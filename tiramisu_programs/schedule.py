import copy
import traceback
import numpy as np
import re
import sys, os, subprocess
from pathlib import Path
from datetime import datetime
from tiramisu_programs.optimization import optimization_command
import time
import re
import torch
import json
from rl_interface.action import Action
from surrogate_model_utils.json_to_tensor import get_tree_structure, get_sched_rep

global_dioph_sols_dict = dict()
EPSILON = 1e-6

class LargeAccessMatices(Exception):
    pass
class NbAccessException(Exception):
    pass
class LoopsDepthException(Exception):
    pass
class TimeOutException(Exception):
    pass
class LoopExtentException(Exception):
    pass
class RepresentationLengthException(Exception):
    pass
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.flatten().tolist()
        return json.JSONEncoder.default(self, obj)
class LCException(Exception):
    pass
class SkewParamsException(Exception):
    pass
class IsTiledException(Exception):
    pass
class IsInterchangedException(Exception):
    pass
class IsSkewedException(Exception):
    pass
class IsUnrolledException(Exception):
    pass
class IsParallelizedException(Exception):
    pass
class IsReversedException(Exception):
    pass
class SkewUnrollException(Exception):
    pass

class Schedule:
    MAX_DEPTH = 6
    def __init__(self, program,model=None,**args):
        self.depth = 0
        self.schedule = []
        self.schedule_str = ""
        self.speedup = 0
        self.is_interchaged=False
        self.is_tiled=False
        self.is_unrolled=False
        self.is_skewed=False
        self.is_parallelized=False
        self.is_reversed=False
        self.model = model
        self.prog = program
        self.args = args
        if "env_type" in self.args:
            if self.args["env_type"] == "cpu":
                self.measurement_env = self.prog.evaluate_schedule
            else:
                self.measurement_env = self.get_exec_time_by_model
        self.comps = list(self.prog.comp_name)
        self.annotations=self.prog.get_program_annotations()
        self.obs = None

    @classmethod
    def linear_diophantine_default(cls,f_i, f_j):
        found = False
        gamma = 0
        sigma = 1
        if (f_j == 1) or (f_i == 1):
            gamma = f_i - 1
            sigma = 1
        else:
            if (f_j == -1) and (f_i > 1):
                gamma = 1
                sigma = 0
            else:
                i = 0
                while (i < 100) and (not found):
                    if ((sigma * f_i) % abs(f_j)) == 1:
                        found = True
                    else:
                        sigma += 1
                        i += 1
                if not found:
                    print("Error cannof find solution to diophantine equation")
                    return
                gamma = ((sigma * f_i) - 1) / f_j

        return gamma, sigma

    @classmethod
    def pad_access_matrix(cls,access_matrix, max_depth):
        access_matrix = np.array(access_matrix)
        access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix] # adding tags for marking the used rows
        access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix] # adding tags for marking the used columns
        padded_access_matrix = np.zeros((max_depth + 1, max_depth + 2))
        padded_access_matrix[:access_matrix.shape[0],:access_matrix.shape[1]-1] = access_matrix[:,:-1] #adding padding to the access matrix before the last column
        padded_access_matrix[:access_matrix.shape[0],-1] = access_matrix[:,-1] #appending the last columns
        
        return padded_access_matrix

    @classmethod
    def isl_to_write_matrix(cls,isl_map): # for now this function only support reductions
        comp_iterators_str = re.findall(r'\[(.*)\]\s*->', isl_map)[0]
        buffer_iterators_str = re.findall(r'->\s*\w*\[(.*)\]', isl_map)[0]
        buffer_iterators_str=re.sub(r"\w+'\s=","",buffer_iterators_str)
        comp_iter_names = re.findall(r'(?:\s*(\w+))+', comp_iterators_str)
        buf_iter_names = re.findall(r'(?:\s*(\w+))+', buffer_iterators_str)
        matrix = np.zeros([len(buf_iter_names),len(comp_iter_names)+1])
        for i,buf_iter in enumerate(buf_iter_names):
            for j,comp_iter in enumerate(comp_iter_names):
                if buf_iter==comp_iter:
                    matrix[i,j]=1
                    break
        return matrix

    @classmethod
    def sched_json_to_sched_str(cls,sched_json, prog_it): # Works only for 1 comp programs
        orig_loop_nest = []
        orig_loop_nest.append(list(prog_it.keys())[0])
        child_list = prog_it[list(prog_it.keys())[0]]['child_iterators']
        while len(child_list)>0:
            child_loop = prog_it[child_list[0]]
            orig_loop_nest.append(child_list[0])
            child_list = child_loop['child_iterators']
            
        comp_name = [n for n in sched_json.keys() if not n in ['unfuse_iterators','tree_structure','execution_times']][0]
        schedule = sched_json[comp_name]
        transf_loop_nest = orig_loop_nest
        sched_str = ''
        
        if schedule['interchange_dims']:
            first_dim_index = transf_loop_nest.index(schedule['interchange_dims'][0])
            second_dim_index = transf_loop_nest.index(schedule['interchange_dims'][1])
            sched_str+='I(L'+str(first_dim_index)+',L'+str(second_dim_index)+')'
            transf_loop_nest[first_dim_index], transf_loop_nest[second_dim_index] = transf_loop_nest[second_dim_index], transf_loop_nest[first_dim_index]
        if schedule['skewing']['skewed_dims']:
            first_dim_index = transf_loop_nest.index(schedule['skewing']['skewed_dims'][0])
            second_dim_index = transf_loop_nest.index(schedule['skewing']['skewed_dims'][1])
            first_factor = schedule['skewing']['skewing_factors'][0]
            second_factor = schedule['skewing']['skewing_factors'][1]
            sched_str+='S(L'+str(first_dim_index)+',L'+str(second_dim_index)+','+str(first_factor)+','+str(second_factor)+')'
        if schedule['parallelized_dim']:
            dim_index = transf_loop_nest.index(schedule['parallelized_dim'])
            sched_str+='P(L'+str(dim_index)+')'
        if schedule['tiling']['tiling_dims']:
            if schedule['tiling']['tiling_depth']==2:
                first_dim = schedule['tiling']['tiling_dims'][0]
                second_dim = schedule['tiling']['tiling_dims'][1]

                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                first_factor = schedule['tiling']['tiling_factors'][0]
                second_factor = schedule['tiling']['tiling_factors'][1]
                sched_str+='T2(L'+str(first_dim_index)+',L'+str(second_dim_index)+','+str(first_factor)+','+str(second_factor)+')'
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[i:i+1]=first_dim+'_outer', second_dim+'_outer'
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[i:i+1]=first_dim+'_inner', second_dim+'_inner'
            else: #tiling depth == 3
                first_dim = schedule['tiling']['tiling_dims'][0]
                second_dim = schedule['tiling']['tiling_dims'][1]
                third_dim = schedule['tiling']['tiling_dims'][2]
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                third_dim_index = transf_loop_nest.index(third_dim)
                first_factor = schedule['tiling']['tiling_factors'][0]
                second_factor = schedule['tiling']['tiling_factors'][1]
                third_factor = schedule['tiling']['tiling_factors'][2]
                sched_str+='T3(L'+str(first_dim_index)+',L'+str(second_dim_index)+',L'+str(third_dim_index)+','+str(first_factor)+','+str(second_factor)+','+str(third_factor)+')'
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[i:i+1]=first_dim+'_outer', second_dim+'_outer', third_dim+'_outer'
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[i:i+1]=first_dim+'_inner', second_dim+'_inner', third_dim+'_inner'
                transf_loop_nest.remove(third_dim)
        if schedule['unrolling_factor']:
            dim_index = len(transf_loop_nest)-1
            dim_name =transf_loop_nest[-1]
            sched_str+='U(L'+str(dim_index)+','+str(schedule['unrolling_factor'])+')'
            transf_loop_nest[dim_index:dim_index+1] = dim_name+'_Uouter', dim_name+'_Uinner'
        if schedule["reversed_dim"]:
            dim_index = transf_loop_nest.index(schedule["reversed_dim"])
            sched_str+='R(L'+str(dim_index)+')'
        
        return sched_str

    @classmethod
    def get_schedules_str(cls,programs_list,programs_dict):
        if programs_dict != {}:
        
            functions_set={}#a dict containing all existed programs in the dataset with their evaluated schedules

            for fun in programs_list: 
                #print(programs_dict[fun]['schedules_list'])
                if 'schedules_list' in programs_dict[fun].keys():
                    schedules=programs_dict[fun]['schedules_list']#[:2]
                    #print(schedules)
                    schedules_set={}#schedules_program_x 

                    
                    for schedule in schedules:
                        #schedule_str = sched_json_to_sched_str(schedule, prog_it) 
                        comp=list(schedule.keys())[0] #we have only one computation
                        schedule_str = schedule[comp]["schedule_str"]    
                        schedules_set[schedule_str]=schedule[comp]["execution_times"]

                    functions_set[fun]=schedules_set
                #schedules_set.append(schedules_subset)#appending schedules_program_x to schedules_set
                
            return functions_set
        else:
            return {}

    @classmethod
    def get_representation(cls,program_annot):
        max_dims= 7
        max_depth=5
        max_accesses = 21 # TODO: check if 10 is enough
        program_representation = []
        indices_dict = dict()
        computations_dict = program_annot['computations']
        ordered_comp_list = sorted(list(computations_dict.keys()), key = lambda x: computations_dict[x]['absolute_order'])

        placeholders_comp = {}
        
        for index, comp_name in enumerate(ordered_comp_list):
            comp_dict = computations_dict[comp_name]
            comp_representation = []
    
            iterators_repr = [] 
            for iter_i,iterator_name in enumerate(comp_dict['iterators']):
                iterator_dict = program_annot['iterators'][iterator_name]
                iterators_repr.extend([iterator_dict['lower_bound'], iterator_dict['upper_bound']])
            
                # transformations placeholders
                l_code='L'+iterator_name
                iterators_repr.extend([l_code+'Interchanged', 
                                    l_code+'Skewed', l_code+'SkewFactor', 
                                    l_code+'Parallelized',
                                    l_code+'Tiled', l_code+'TileFactor',
                                    l_code+'Reversed',
                                    l_code+'Fused',
                                    0, 
                                    0,
                                    l_code+"_1"+'Interchanged',
                                    l_code+"_1"+'Skewed', l_code+"_1"+'SkewFactor',
                                    l_code+"_1"+'Parallelized',
                                    l_code+"_1"+'Tiled', l_code+"_1"+'TileFactor',
                                    l_code+"_1"+'Reversed',
                                    l_code+"_1"+'Fused']) #unrolling is skipped since it is added only once
            
            # Adding padding
            iterator_repr_size = int(len(iterators_repr)/(2*len(comp_dict['iterators'])))
            iterators_repr.extend([0]*iterator_repr_size*2*(max_depth-len(comp_dict['iterators']))) # adding iterators padding 

            # Adding unrolling placeholder since unrolling can only be applied to the innermost loop 
            iterators_repr.extend(['Unrolled', 'UnrollFactor'])
            
            # Adding the iterators representation to computation vector       
            comp_representation.extend(iterators_repr)
            
            #  Write access representation to computation vector
            padded_write_matrix = cls.pad_access_matrix(cls.isl_to_write_matrix(comp_dict['write_access_relation']), max_depth)
            write_access_repr = [comp_dict['write_buffer_id']+1] + padded_write_matrix.flatten().tolist() # buffer_id + flattened access matrix 

        #     print('write ', comp_dict['write_buffer_id']+1,'\n',padded_write_matrix)
            
            # Adding write access representation to computation vector
            comp_representation.extend(write_access_repr)
            
            # Read Access representation 
            read_accesses_repr=[]
            for read_access_dict in comp_dict['accesses']:
                read_access_matrix = cls.pad_access_matrix(read_access_dict['access_matrix'], max_depth)
                read_access_repr = [read_access_dict['buffer_id']+1] + read_access_matrix.flatten().tolist() # buffer_id + flattened access matrix 
                read_accesses_repr.extend(read_access_repr)
        #         print('read ', read_access_dict['buffer_id']+1,'\n',read_access_matrix)

                
            access_repr_len = (max_depth+1)*(max_depth + 2) + 1 # access matrix size +1 for buffer id
            read_accesses_repr.extend([0]*access_repr_len*(max_accesses-len(comp_dict['accesses']))) #adding accesses padding


            # Adding read Accesses to the representation to computation vector
            comp_representation.extend(read_accesses_repr)
            
            # Adding Operations count to computation vector
            comp_representation.append(comp_dict['number_of_additions'])
            comp_representation.append(comp_dict['number_of_subtraction'])
            comp_representation.append(comp_dict['number_of_multiplication'])
            comp_representation.append(comp_dict['number_of_division'])

            #print("comp rep before placeholders", comp_representation)



            placeholders_indices_dict = {}
            for i, element in enumerate(comp_representation):
                if isinstance(element, str):
                    placeholders_indices_dict[element] = i
                    comp_representation[i]=0
            placeholders_comp[comp_name]= placeholders_indices_dict
            


            # adding log(x+1) of the representation
            # log_rep = list(np.log1p(comp_representation))
            # comp_representation.extend(log_rep)
            
            program_representation.append(comp_representation)
            indices_dict[comp_name] = index
        
        return program_representation, placeholders_comp, indices_dict

    @classmethod
    def get_representation_template(cls,program_annot):
        # print("in repr template")
        max_accesses = 15
        min_accesses = 1
        max_depth = 5 

        comp_name = list(program_annot['computations'].keys())[0] # for single comp programs, there is only one computation
        comp_dict = program_annot['computations'][comp_name] 
        
        if len(comp_dict['accesses'])>max_accesses:
            raise NbAccessException
        if len(comp_dict['accesses'])<min_accesses:
            raise NbAccessException
        if len(comp_dict['iterators'])>max_depth:
            raise LoopsDepthException

        
        comp_repr_template = []
    #         iterators representation + transformations placeholders
        iterators_repr = []    
        for iter_i,iterator_name in enumerate(comp_dict['iterators']):
            iterator_dict = program_annot['iterators'][iterator_name]
            iterators_repr.extend([iterator_dict['lower_bound'], iterator_dict['upper_bound']])
        
            # transformations placeholders
            l_code='L'+iterator_name
            iterators_repr.extend([l_code+'Interchanged', 
                                l_code+'Skewed', l_code+'SkewFactor', 
                                l_code+'Parallelized',
                                l_code+'Tiled', l_code+'TileFactor',
                                l_code+'Reversed',
                                0, 
                                0,
                                l_code+"_1"+'Interchanged',
                                l_code+"_1"+'Skewed', l_code+"_1"+'SkewFactor',
                                l_code+"_1"+'Parallelized',
                                l_code+"_1"+'Tiled', l_code+'TileFactor',
                                l_code+"_1"+'Reversed']) #unrolling is skipped since it is added only once
        
        # Adding padding
        iterator_repr_size = int(len(iterators_repr)/(2*len(comp_dict['iterators'])))
        iterators_repr.extend([0]*iterator_repr_size*2*(max_depth-len(comp_dict['iterators']))) # adding iterators padding 

        # Adding unrolling placeholder since unrolling can only be applied to the innermost loop 
        iterators_repr.extend(['Unrolled', 'UnrollFactor'])
        
        # Adding the iterators representation to computation vector       
        comp_repr_template.extend(iterators_repr)
        
        #  Write access representation to computation vector
        padded_write_matrix = cls.pad_access_matrix(cls.isl_to_write_matrix(comp_dict['write_access_relation']), max_depth)
        write_access_repr = [comp_dict['write_buffer_id']+1] + padded_write_matrix.flatten().tolist() # buffer_id + flattened access matrix 

    #     # print('write ', comp_dict['write_buffer_id']+1,'\n',padded_write_matrix)
        
        # Adding write access representation to computation vector
        comp_repr_template.extend(write_access_repr)
        
        # Read Access representation 
        read_accesses_repr=[]
        for read_access_dict in comp_dict['accesses']:
            read_access_matrix = cls.pad_access_matrix(read_access_dict['access_matrix'], max_depth)
            read_access_repr = [read_access_dict['buffer_id']+1] + read_access_matrix.flatten().tolist() # buffer_id + flattened access matrix 
            read_accesses_repr.extend(read_access_repr)
    #         # print('read ', read_access_dict['buffer_id']+1,'\n',read_access_matrix)

            
        access_repr_len = (max_depth+1)*(max_depth + 2) + 1 # access matrix size +1 for buffer id
        read_accesses_repr.extend([0]*access_repr_len*(max_accesses-len(comp_dict['accesses']))) #adding accesses padding


        # Adding read Accesses to the representation to computation vector
        comp_repr_template.extend(read_accesses_repr)
        
        # Adding Operations count to computation vector
        comp_repr_template.append(comp_dict['number_of_additions'])
        comp_repr_template.append(comp_dict['number_of_subtraction'])
        comp_repr_template.append(comp_dict['number_of_multiplication'])
        comp_repr_template.append(comp_dict['number_of_division'])
        
        # Track the indices to the placeholders in a a dict
        placeholders_indices_dict = {}
        for i, element in enumerate(comp_repr_template):
            if isinstance(element, str):
                placeholders_indices_dict[element] = i
                comp_repr_template[i]=0

        
        return comp_repr_template, placeholders_indices_dict

    @classmethod
    def sched_str(cls,sched_str, id, params, comp_indic):
        if id in range(28):
            sched_str+='I(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+')'
        else:
            if id in range(28, 41):
                if params["tiling_depth"]==2:
                    sched_str+='T2(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+','+str(params["first_factor"])+','+str(params["second_factor"])+')'
                else:
                    sched_str+='T3(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+',L'+str(params["third_dim_index"])+','+str(params["first_factor"])+','+str(params["second_factor"])+','+str(params["third_factor"])+')'
            else:
                if id in range(41,44):
                    for comp in params:
                        sched_str+='U(L'+str(params[comp]["dim_index"])+','+str(params[comp]['unrolling_factor'])+ ",C"+ str(comp_indic[comp])+')'
                else:
                    if id in range(44,46):
                        sched_str+='S(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+','+str(params["first_factor"])+','+str(params["second_factor"])+')'
                    else:
                        if id in range(46,48):
                            sched_str+='P(L'+str(params["dim_index"])+')'
                        else:
                            if id in range(48,56):
                                sched_str+='R(L'+str(params["dim_index"])+')'
                            else:
                                if id in range(56,61):
                                    sched_str+='F(L'+str(params["dim_index"])+')'

        return sched_str

    @classmethod
    def get_orig_tree_struct(cls,program_json,root_iterator):
        tree_struct = {'loop_name':root_iterator,'computations_list':program_json['iterators'][root_iterator]['computations_list'][:],'child_list':[]}
        for child_iterator in program_json['iterators'][root_iterator]['child_iterators']:
            tree_struct['child_list'].append(self.get_orig_tree_struct(program_json,child_iterator))
        return tree_struct

    def get_observation(self):
        if self.obs is not None: return self.obs 
        self.prog_rep, self.comps_placeholders, self.comp_indic_dict = self.get_representation(self.annotations)

        # print("the length is", len(prog_rep[0]))

        for comp_rep in self.prog_rep:
            if len(comp_rep) != 1052:
                raise RepresentationLengthException
        
        
        if len(self.comps)!= 1:
            # print("more than one comp")
            self.comps_it = []
            for comp in self.comps:
                self.comps_it.append(self.annotations["computations"][comp]["iterators"])
            
            ## print("got the comp it", self.comps_it)

            self.common_it = self.comps_it[0]

            for comp_it in self.comps_it[1:]:
                ## print("common it is ", self.common_it)
                self.common_it = [it for it in comp_it if it in self.common_it]

            # print("the common iterators are", self.common_it)

        elif len(self.comps)>5: # To avoid IndexError in self.obs["representation"]
            raise IndexError

        else:
            # print("one comp, no need for common iterators")
            self.common_it= self.annotations["computations"][self.comps[0]]["iterators"]


        # print("The initial execution time is", self.prog.initial_execution_time)
        self.schedule_dict = dict()
        self.schedule_dict["fusions"] = None
        for comp in self.comps:
            dim = len(self.annotations['computations'][comp]['iterators'])
            self.schedule_dict[comp] = dict()
            self.schedule_dict[comp]["dim"] = dim
            self.schedule_dict[comp]["transformation_matrix"] = np.eye(dim,dim)
            self.schedule_dict[comp]["transformation_matrices"] = [np.eye(dim,dim)]
            self.schedule_dict[comp]['parallelized_dim'] = None
            self.schedule_dict[comp]['unrolling_factor'] = None
            self.schedule_dict[comp]['tiling'] = None
        self.schedule_dict['tree_structure'] = get_tree_structure(self.annotations)
        
        self.templates = dict()
        (self.templates["prog_tree"],
            self.templates["comps_repr_templates_list"],
            self.templates["loops_repr_templates_list"],
            self.templates["comps_placeholders_indices_dict"],
            self.templates["loops_placeholders_indices_dict"]) = get_sched_rep(self.annotations, self.schedule_dict, max_depth=self.MAX_DEPTH-1)
        self.schedule_dict["fusions"] = []
        self.placeholders = self.comps_placeholders
        self.added_iterators=[]   

        self.obs={}
        self.obs["representation"] = np.empty((0,1052),np.float32)
        self.obs["loops_representation"]=np.empty((0,26),np.float32)
        self.obs['child_list']=np.empty((0,11),np.float32)
        self.obs['has_comps']=np.empty((0,12),np.float32)
        self.obs['computations_indices']=np.empty((0,5),np.float32)

        for i in range (5):
            if i>=len(self.prog_rep):
                self.obs["representation"]=np.vstack([self.obs["representation"], np.zeros(1052)])
            else:
                self.obs["representation"]=np.vstack([self.obs["representation"], np.array([self.prog_rep[i]],dtype=np.float32)])

        #print("\nLa représentation vectorielle initiale de ce programme est:", self.obs["representation"] )
        
        print("\nLes niveaux de boucles de ce programme sont:")
        self.it_dict={}
        for comp in self.comps:        
            comp_it_dict={}
            iterators=list(self.annotations["computations"][comp]["iterators"])
            
            for i in range (len(iterators)):
                comp_it_dict[i]={}
                comp_it_dict[i]['iterator']=iterators[i]
                comp_it_dict[i]['lower_bound']=self.annotations['iterators'][iterators[i]]['lower_bound']
                comp_it_dict[i]['upper_bound']=self.annotations['iterators'][iterators[i]]['upper_bound']

            self.it_dict[comp]=comp_it_dict
        print(self.it_dict)

        iterators=list(self.annotations["iterators"].keys())

        for i in range(len(iterators)):
        
            loop_repr=[]
            loop_repr.append(self.annotations['iterators'][iterators[i]]['lower_bound'])
            loop_repr.append(self.annotations['iterators'][iterators[i]]['upper_bound'])
            loop_repr.extend([0,0,0,0,0,0,0,0,0,0,0])
            loop_log_rep = list(np.log1p(loop_repr))
            loop_repr.extend(loop_log_rep)
            self.obs["loops_representation"]=np.vstack([self.obs["loops_representation"],np.array([loop_repr])])

            childs_indexes=[iterators.index(child) for child in self.annotations['iterators'][iterators[i]]['child_iterators']]
            if len(childs_indexes)!=11:
                for j in range(11-len(childs_indexes)):
                    childs_indexes.append(-1)
            self.obs["child_list"]=np.vstack([self.obs["child_list"], np.array([childs_indexes])])
            
            if self.annotations['iterators'][iterators[i]]['computations_list']!=[]:
                self.obs['has_comps']=np.append(self.obs['has_comps'],1)
            else:
                self.obs['has_comps']=np.append(self.obs['has_comps'],0)

            computations_list=list(self.annotations['computations'].keys())
            loop_comps=[computations_list.index(comp) for comp in self.annotations['iterators'][iterators[i]]['computations_list']]
            if len(loop_comps)!=5:
                for j in range(5-len(loop_comps)):
                    loop_comps.append(-1)
            self.obs["computations_indices"]=np.vstack([self.obs["computations_indices"],np.array([loop_comps])])
        

        #Add null vectors if needed to avoid mismatching error of env.observation's type and reset_obs's type              
        for i in range(15-len(self.annotations["iterators"])):
            loop_repr=np.full(26,-1)
            self.obs["loops_representation"]=np.vstack([self.obs["loops_representation"],loop_repr])
        
        for i in range(12-len(self.annotations["iterators"])):
            self.obs["child_list"]=np.vstack([self.obs["child_list"], np.full(11,-1)])
            self.obs['has_comps']=np.append(self.obs['has_comps'],0)
            self.obs["computations_indices"]=np.vstack([self.obs["computations_indices"],np.full(5,-1)])

        
        if len(self.common_it) == 5:
            self.obs["action_mask"] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        else:
            if len(self.common_it) == 4:
                self.obs["action_mask"] = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
            else: 
                if len(self.common_it) == 3:
                    self.obs["action_mask"] = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
                else: 
                    if len(self.common_it) == 2:
                        self.obs["action_mask"] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
                    else:
                        if len(self.common_it) == 1:
                            self.obs["action_mask"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    
        if len(self.comps)==1:
            np.put(self.obs["action_mask"],[56,57,58,59,60],[0, 0, 0, 0, 0])  
        return self.obs

    def apply_action(self,action):
        first_comp = self.comps[0]
        if not action.id in range(44,46):
            action_params = action.parameter()
            # print("action params first are", action_params)
        else:
            comp=list(self.it_dict.keys())[0]
            action_params=action.parameter(comp, self.prog)
        if action.id in range(28):
            
            if not self.is_interchaged:
                
                params=[int(action_params["first_dim_index"]), int(action_params["second_dim_index"])]

                optim1 = optimization_command("Interchange", params, self.comps)
                print("got the optim cmd")
                self.schedule.append(optim1)
        
                
                if self.is_unrolled:
                    lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                else:
                    lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)
                
                # print("\n in interchange,  lc res: {}".format(lc_check))
                            


                if lc_check == -1: 
                    print("\nCette action a généré une erreur")
                    self.obs["action_mask"][action.id]=0    
                    raise LCException

                if lc_check == 0:
                    print("\nCette action est illégale")
                    self.schedule.pop()
                    info = {"illegal_action": True}
                    done = False
                    return self.obs, reward, done, info
                
                self.apply_interchange(action_params)
                print("interchange applied")
                self.is_interchaged=True
                
            else:
                print("interchange already applied execption")
                applied_exception=True
                raise IsInterchangedException
                #to expierment with the reward in this case
                
        if action.id in range(28,41):
            if not self.is_tiled:
                params=[int(action_params["first_dim_index"]), int(action_params["second_dim_index"])]
                params.append(action_params["first_factor"])
                params.append(action_params["second_factor"])
                
                if action_params["tiling_depth"] == 3:
                    params.insert(2, action_params["third_dim_index"])
                    params.append(action_params["third_factor"])

                
                optim2 = optimization_command("Tiling", params, self.comps)

                self.schedule.append(optim2)


                if self.is_unrolled:
                    lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                else:
                    lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)

                
                print("\n in tiling,  lc res: {}".format(lc_check))
                
                
                if lc_check == -1:   
                    print("\nCette action a généré une erreur")
                    raise LCException

                if lc_check == 0:
                    print("\nCette action est illégale")
                    self.schedule.pop()
                    info = {"illegal_action": True}
                    done = False
                    return self.obs, reward, done, info

                self.apply_tiling(action_params)
                print("\n tiling applied")

                self.is_tiled=True
                
                # done = True
                # exit = True
                # self.schedule_str = sched_str(self.schedule_str, action.id, action_params, self.comp_indic_dict)
            else:
                print("\n tiling already applied execption")
                applied_exception=True
                raise IsTiledException

        if action.id in range(41,44):
            params = {}
            if not self.is_unrolled:
                # print("action params of unrolling", action_params["dim_index"])
                # print("action params of unrolling", action_params["unrolling_factor"])

                #we don't apply unrolling on a level that's skewed, we get the tag to see if it's skewed or not
                self.non_skewed_comps = []
                for comp in self.comps:
                    it_skewed="L"+self.it_dict[comp][action_params[comp]["dim_index"]]["iterator"]+"Skewed"
                    if self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][it_skewed]]!=1:
                        self.non_skewed_comps.append(comp)
                
                #for mult comps, unrolling returns a dict of parameters, each for each comp
                for comp in self.non_skewed_comps:
                    params[comp]=[int(action_params[comp]["dim_index"]), int(action_params[comp]["unrolling_factor"])]
                print("\nLes paramètres sont:",params)

                if self.non_skewed_comps != []:
                    print("it's not skewed")

                    optim3 = optimization_command( "Unrolling", params, self.non_skewed_comps)
                    print("obtained tiramisu code")
                    self.schedule.append(optim3)

                    start_time = time.time()
                    
                    lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                    l_time = time.time() - start_time
                    print("\n unrollling lc check {} ".format(lc_check))
                    self.lc_total_time+=l_time
                    
                
                    if lc_check == -1: 
                        print("\nCette action a généré une erreur")                          
                        raise LCException

                    if lc_check == 0:
                        print("\nCette action est illégale")
                        self.schedule.pop()
                        #reward = -1
                        info = {"illegal_action": True}
                        done = False
                        return self.obs, reward, done, info

                    #self.apply_unrolling(action_params)
                    print("\n unrolling applied")
                    for i in range(41,44):
                        self.obs["action_mask"][i]=0
                    self.is_unrolled=True
                else:
                    #reward=-1
                    lc_check=0
                    info['error']="trying to apply unrolling after skewing in one of the computations"
                
            else:
                applied_exception=True
                print("\n unrolling is already applied")

                raise IsUnrolledException

        if action.id in range(44,46):

            if not self.is_skewed:

            
                if (action_params["first_factor"] != None and action_params["second_factor"] != None):
                    
                    print("\nLes paramètres sont:")
                    print("\nLe premier niveau de boucle:", action_params["first_dim_index"])
                    print("\nLe deuxième niveau de boucle:", action_params["second_dim_index"])
                    print("\nLe premier facteur:", action_params["first_factor"])
                    print("\nLe deuxième facteur:", action_params["second_factor"])
                    non_inner_comps = []
                    for comp in self.comps:
                        if (action_params["first_dim_index"] != len(self.it_dict[comp])-1 and action_params["second_dim_index"] != len(self.it_dict[comp])-1) or ( (action_params["first_dim_index"] == len(self.it_dict[comp])-1 or action_params["second_dim_index"] == len(self.it_dict[comp])-1 and not self.is_unrolled )) :
                            non_inner_comps.append(comp)


                    if non_inner_comps != []:

                        params=[int(action_params["first_dim_index"]), int(action_params["second_dim_index"])]
                        params.append(action_params["first_factor"])
                        params.append(action_params["second_factor"])

                        optim4 = optimization_command("Skewing", params, non_inner_comps)

                        self.schedule.append(optim4)

                        start_time = time.time()
                        if self.is_unrolled:
                            lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                        else:
                            lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)
                        l_time = time.time() - start_time
                        print("\n skewing lc check res {} ".format(lc_check))
                        self.lc_total_time+=l_time

                    
                        if lc_check == -1:   
                            print("\nCette action a généré une erreur")
                            raise LCException
                        if lc_check == 0:
                            print("\nCette action est illégale")
                            self.schedule.pop()
                            #reward = -1
                            info = {"illegal_action": True}
                            done = False
                            return self.obs, reward, done, info

                        self.apply_skewing(action_params)
                        print("\n skewing is applied")
                        self.is_skewed=True

                    else:
                        skew_unroll=True
                        raise SkewUnrollException

                else:
                    print("\n skewing prams are null")
                    skew_params_exception=True
                    raise SkewParamsException

            
            
            else:
                print("\n sekwing is already applied")
                applied_exception=True
                raise IsSkewedException

        if action.id in range(46,48):
            if not self.is_parallelized:
                print("\nLes paramètres sont:")
                print("\nLe niveau de boucle:", action_params["dim_index"])

                params=[int(action_params["dim_index"])]

                optim5 = optimization_command("Parallelization", params, self.comps)
            
                self.schedule.append(optim5)

                start_time = time.time()
                if self.is_unrolled:
                    lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                else:
                    lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)
                
                l_time = time.time() - start_time
                print("\n parallelzation lc check {}".format(lc_check))
                self.lc_total_time+=l_time

                if lc_check == -1:    
                    print("\nCette action a généré une erreur")   
                    raise LCException

                if lc_check == 0:
                    print("\nCette action est illégale")
                    self.schedule.pop()
                    #reward = -1
                    info = {"illegal_action": True}
                    done = False
                    return self.obs, reward, done, info

                self.apply_parallelization(action_params)
                print("\n parallelisation applied")
                self.is_parallelized=True
            else:
                applied_exception=True
                print("\n parallelisation is already applied")
                raise IsParallelizedException

        if action.id in range(48,56):
            
            if not self.is_reversed:
                print("\nLes paramètres sont:")
                print("\nLe niveau de boucle:", action_params["dim_index"])

                params=[int(action_params["dim_index"])]

                optim6 = optimization_command( "Reversal", params, self.comps)

                self.schedule.append(optim6)
                
                start_time=time.time()
                if self.is_unrolled:
                    lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                else:
                    lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)
                l_time = time.time() - start_time
                print("loop reversal lc check {}".format(lc_check))
                self.lc_total_time+=l_time

                if lc_check == -1: 
                    print("\nCette action a généré une erreur")
                    self.obs["action_mask"][action.id]=0
                    raise LCException

                if lc_check == 0:
                    print("\nCette action est illégale")
                    self.schedule.pop()
                    #self.obs["action_mask"][action.id]=0
                    #reward = -1
                    info = {"illegal_action": True}
                    done = False
                    return self.obs, reward, done, info

                self.apply_reversal(action_params)
                print("\n loop reversal applied")
                self.is_reversed=True
            else:
                applied_exception=True

                print("\n loop reversal already applied")

                raise IsReversedException

        if action.id in range(56,61):
            params=[int(action_params["dim_index"]), action_params["fuse_comps"]]

            print("fuse params are", action_params["dim_index"], '\n', action_params["fuse_comps"])

            if action_params["fuse_comps"] != [] and len(action_params["fuse_comps"])!=1:

                optim7 = optimization_command( "Fusion", params, action_params["fuse_comps"])

                print("fusion optim created")

                self.schedule.append(optim7)
                
                start_time=time.time()

                if self.is_unrolled:
                    lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                else:
                    lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)
                    
                l_time = time.time() - start_time
                print("loop fusion lc check {}".format(lc_check))
                self.lc_total_time+=l_time

                if lc_check == -1: 
                    print("\nCette action a généré une erreur")
                    self.obs["action_mask"][action.id]=0
                    raise LCException

                if lc_check == 0:
                    print("\nCette action est illégale")
                    self.schedule.pop()
                    info = {"illegal_action": True}
                    done = False
                    return self.obs, reward, done, info

                self.apply_fusion(action_params)
                print("\n loop fusion applied")
                self.is_fused=True
            else:
                lc_check=0
                print("unable to fuse")
                #reward=-1


        if action.id==Action.EXIT:
            print("\n **** it's an exit action ****")
            done=True
            exit=True
            
        if (not exit and lc_check!=0) and not (action.id in range(41,44) and self.is_skewed):
            # print("in the long cond after actions")
            self.schedule_str = sched_str(self.schedule_str, action.id, action_params, self.comp_indic_dict)
            # print("the original iterators were:", self.it_dict)
            if not action.id in range(41,44):
                self.it_dict=update_iterators(action.id, self.it_dict, action_params, self.added_iterators, self.comp_indic_dict)
                print("after update iterators with the schedule", self.schedule_str, "it is", self.it_dict)

            self.depth += 1
        
        if(self.depth == self.MAX_DEPTH) or (self.steps >=20):
            done=True

        # print("--- done is ----", done)

        if done: 
            print("\nFin de l'épisode")
            if self.is_unrolled:       
                for optim in self.schedule:
                    print(optim.type)
                    if optim.type == "Unrolling":
                        unroll_optimisation=optim

                new_unrolling_params={}
                new_unrolling_optim_params={}
                for comp in self.non_skewed_comps:
                    unroll_factor=unroll_optimisation.params_list[comp][1]
                    new_unrolling_params[comp]={"dim_index":len(self.it_dict[comp])-1,"unrolling_factor":unroll_factor}
                    new_unrolling_optim_params[comp]=[len(self.it_dict[comp])-1, unroll_factor]
                    
                new_unrolling_optim=optimization_command("Unrolling", new_unrolling_optim_params, self.non_skewed_comps)
                print("Done")
                new_unrolling_str=""
                unrolling_str=""

                print("1")
                for comp in self.non_skewed_comps: 
                    unroll_factor=unroll_optimisation.params_list[comp][1]
                    print("1.1")
                    # print("comp", comp)  
                    # print("unroll_factor", unroll_factor)
                    new_unrolling_str+="U(L"+str(len(self.it_dict[comp])-1)+","+str(unroll_factor)+",C"+str(self.comp_indic_dict[comp]) +")"
                    print("1.2")
                    #print("new_unrolling_str",new_unrolling_str)
                    unrolling_str+="U(L"+str(unroll_optimisation.params_list[comp][0])+","+str(unroll_factor)+",C"+str(self.comp_indic_dict[comp]) +")" 
                    print("1.3")
                    #print("unrolling_str", unrolling_str)

                self.schedule_str=self.schedule_str.replace(unrolling_str, "") + new_unrolling_str
                print("1.4")
                self.schedule.remove(unroll_optimisation)      
                self.schedule.append(new_unrolling_optim)
                print("1.5")
                self.apply_unrolling(new_unrolling_params)
                print("2")
                #no need to update the iterators list because it's the end of the episode



            self.search_time= time.time()-self.search_time
            
            try:
                exec_time=0
                writing_time=0
                exec_time = self.get_exec_time()

                if not self.is_parallelized:
                    #print("inside parallelization in done")
                    print("Tester si la parallélisation apporte un meilleur speedup...")
                    action = Action(self.PARALLELIZATION0, self.it_dict, self.common_it)
                    action_params = action.parameter()

                    params=[int(action_params["dim_index"])]

                    optim5 = optimization_command("Parallelization", params, self.comps)
                    first_comp=list(self.it_dict.keys())[0]
                    iterator = self.it_dict[first_comp][action_params["dim_index"]]['iterator']
                    self.schedule_dict[first_comp]["parallelized_dim"] = iterator
                
                    self.schedule.append(optim5)


                    try:

                        self.schedule_str = sched_str(self.schedule_str, action.id, action_params, self.comp_indic_dict)
                        parallelized_exec_time=self.get_exec_time()
                        parallelization_str='P(L'+str(action_params["dim_index"])+')'
                        print("exec time with parallelization: ", parallelized_exec_time)
                        
                        # print("the exec time with parallelization is", parallelized_exec_time)
                        # print("the exec time without parallelization is", exec_time)
                    except:
                        print("\nCette action est illégale")
                        self.schedule.remove(optim5)
                        self.schedule_str=self.schedule_str.replace(parallelization_str, "")
                        
                    
                    if parallelized_exec_time < exec_time and parallelized_exec_time!=0:
                        exec_time = parallelized_exec_time
                        
                        self.apply_parallelization(action_params)
                        print("La parallélisation améliore le temps d'exécution donc elle est appliquée.")

                    else:
                        self.schedule.remove(optim5)
                        self.new_scheds[self.prog.name].pop(self.schedule_str)
                        self.schedule_str=self.schedule_str.replace(parallelization_str, "")
                        self.schedule_dict[first_comp]["parallelized_dim"] = None
                        print("La parallélisation n'améliore pas le temps d'exécution, alors elle n'est pas appliquée.")
                

            except:
                print("\nErreur lors de la mesure du temps d'exécution.") 
                info = {"Internal execution error": True}
                #reward=-1

                print("error with get execution time, going out")
                return self.obs, reward, done, info

            if exec_time!=0:
                print("\nLe schedule final trouvé est: ",self.schedule_str)
                print("The new execution time is ", exec_time)
                #self.speedup = (self.prog.initial_execution_time - exec_time)/self.prog.initial_execution_time
                self.speedup = (self.prog.initial_execution_time / exec_time) + EPSILON
                # if self.prog.initial_execution_time >=  exec_time:
                    
                #     self.speedup = (self.prog.initial_execution_time / exec_time)
                # else:
                #     self.speedup = -(exec_time / self.prog.initial_execution_time )
                
                print("the speedup is: ", self.speedup)
                reward=math.log(self.speedup,2)
                print('the new scheds are', self.new_scheds)
                start_time=time.time()
                try:
                    self.save_sched_to_dataset()
                    self.write_data()
                    writing_time=time.time()-start_time
                    print("Data saved in ",writing_time)
                except:
                    print(f"failed to save schedule", traceback.format_exc() , file=sys.stderr, flush=True)
                    # print("failed to save schedule")

            self.episode_total_time= time.time()-self.episode_total_time
            # print("CODE GEN :",self.codegen_total_time)
            # print("LC : ",self.lc_total_time)
            print("\nEPISODE TOTAL TIME : {}\nLEGALITY CHECK TIME RATIO : {}\nCODE GENERATION TIME RATIO : {}\nWRITING TIME RATIO : {}\n".format(self.episode_total_time, self.lc_total_time/self.episode_total_time, self.codegen_total_time/self.episode_total_time, writing_time/self.episode_total_time))

        info["depth"] =  self.depth


        print("the reward is",reward)          
    
    def update_iterators(self,id, it_list, action_params, added_iterators, comp_indic_dict):
        for comp in it_list:
            if id in range(28):
                tmp=it_list[comp][action_params["first_dim_index"]]
                it_list[comp][action_params["first_dim_index"]]=it_list[comp].pop(action_params["second_dim_index"])
                it_list[comp][action_params["second_dim_index"]]=tmp

            if id in range(28,41):
                depth_1=action_params["first_dim_index"]
                depth_2=action_params["second_dim_index"]

                keys=list(it_list[comp].keys())
                print("keys: ", keys)

                i=len(keys)-1

                
                if action_params["tiling_depth"]==2:
                    while i>depth_2:
                        if action_params["tiling_loop_1"] and action_params["tiling_loop_2"]:
                                it_list[comp][i+2]=it_list[comp][i]
                        elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"]:   
                                it_list[comp][i+1]=it_list[comp][i]
                        i-=1

                else:
                    if action_params["tiling_depth"]==3:
                        depth_3=action_params["third_dim_index"]
                        print("third depth is", depth_3)
                        while i>depth_3:
                            if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:   
                                it_list[comp][i+3]=it_list[comp][i]
                            else:
                                booleans=[action_params["tiling_loop_1"], action_params["tiling_loop_2"], action_params["tiling_loop_3"]]
                                if booleans.count(True)==2:
                                    it_list[comp][i+2]=it_list[comp][i]
                                elif booleans.count(True)==1:
                                    it_list[comp][i+1]=it_list[comp][i]
                            i-=1
                
                                

                if action_params["tiling_depth"]==2:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"]:
                        # print("in action params == 7 and tiling_loop_1 and tiling_loop_2")


                        #update the loop bounds if tiling is applied on loop 1
                        it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                        it_list[comp][depth_1+2]={}
                        it_list[comp][depth_1+2]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])
                        it_list[comp][depth_1+2]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                        it_list[comp][depth_1+2]['upper_bound']=action_params["first_factor"]
                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_1+2]['iterator'])
                        #update the loop bounds if tiling is applied on loop 2
                        it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                        it_list[comp][depth_2+2]={}
                        it_list[comp][depth_2+2]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                        it_list[comp][depth_2+2]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                        it_list[comp][depth_2+2]['upper_bound']=action_params["second_factor"]
                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_2+2]['iterator'])

                    else:
                        if action_params["tiling_loop_1"]:
                            # print("in action params == 7 and tiling_loop_1")
                            #update the loop bounds if tiling is applied on loop 1
                            it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                            it_list[comp][depth_1+2]={}
                            it_list[comp][depth_1+2]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])
                            it_list[comp][depth_1+2]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                            it_list[comp][depth_1+2]['upper_bound']=action_params["first_factor"]

                            #Add the new iterator to added_iterators
                            added_iterators.append(it_list[comp][depth_1+2]['iterator'])

                        elif action_params["tiling_loop_2"]:
                            # print("in action params == 7 and tiling_loop_2")
                            #update the loop bounds if tiling is applied on loop 2
                            it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                            it_list[comp][depth_2+1]={}
                            it_list[comp][depth_2+1]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                            it_list[comp][depth_2+1]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                            it_list[comp][depth_2+1]['upper_bound']=action_params["second_factor"]

                            #Add the new iterator to added_iterators
                            added_iterators.append(it_list[comp][depth_2+1]['iterator'])

                elif action_params["tiling_depth"]==3:

                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        # print("in action params == 10 and tiling_loop_1 and tiling_loop_2 and tiling_loop_3")

                        #update the loop bounds if tiling is applied on loop 1
                        it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                        it_list[comp][depth_1+3]={}
                        it_list[comp][depth_1+3]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])   
                        it_list[comp][depth_1+3]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                        it_list[comp][depth_1+3]['upper_bound']=action_params["first_factor"]

                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_1+3]['iterator'])

                        #update the loop bounds if tiling is applied on loop 2
                        it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                        it_list[comp][depth_2+3]={}
                        it_list[comp][depth_2+3]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                        it_list[comp][depth_2+3]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                        it_list[comp][depth_2+3]['upper_bound']=action_params["second_factor"]

                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_2+3]['iterator'])

                        #update the loop bounds if tiling is applied on loop 1=3
                        it_list[comp][depth_3]['upper_bound']=it_list[comp][depth_3]['upper_bound']/action_params["third_factor"]             
                        it_list[comp][depth_3+3]={}
                        it_list[comp][depth_3+3]['iterator']="{}_1".format(it_list[comp][depth_3]['iterator'])
                        it_list[comp][depth_3+3]['lower_bound']=it_list[comp][depth_3]['lower_bound']
                        it_list[comp][depth_3+3]['upper_bound']=action_params["third_factor"]

                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_3+3]['iterator'])
                    
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"]:
                        # print("in action params == 10 and tiling_loop_1 and tiling_loop_2")

                        #update the loop bounds if tiling is applied on loop 1
                        it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                        it_list[comp][depth_1+3]={}
                        it_list[comp][depth_1+3]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])   
                        it_list[comp][depth_1+3]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                        it_list[comp][depth_1+3]['upper_bound']=action_params["first_factor"]
                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_1+3]['iterator'])

                        #update the loop bounds if tiling is applied on loop 2
                        it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                        it_list[comp][depth_2+3]={}
                        it_list[comp][depth_2+3]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                        it_list[comp][depth_2+3]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                        it_list[comp][depth_2+3]['upper_bound']=action_params["second_factor"]
                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_2+3]['iterator'])

                    elif action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        # print("in action params == 10 and tiling_loop_2 and tiling_loop_3")

                        #update the loop bounds if tiling is applied on loop 2
                        it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                        it_list[comp][depth_2+2]={}
                        it_list[comp][depth_2+2]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                        it_list[comp][depth_2+2]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                        it_list[comp][depth_2+2]['upper_bound']=action_params["second_factor"]

                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_2+2]['iterator'])

                        #update the loop bounds if tiling is applied on loop 1
                        it_list[comp][depth_3]['upper_bound']=it_list[comp][depth_3]['upper_bound']/action_params["third_factor"]
                        it_list[comp][depth_3+2]={}
                        it_list[comp][depth_3+2]['iterator']="{}_1".format(it_list[comp][depth_3]['iterator'])   
                        it_list[comp][depth_3+2]['lower_bound']=it_list[comp][depth_3]['lower_bound']
                        it_list[comp][depth_3+2]['upper_bound']=action_params["third_factor"]
                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_3+2]['iterator'])

                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        # print("in action params == 10 and tiling_loop_1 and tiling_loop_3")

                        #update the loop bounds if tiling is applied on loop 2
                        it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                        it_list[comp][depth_1+3]={}
                        it_list[comp][depth_1+3]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])
                        it_list[comp][depth_1+3]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                        it_list[comp][depth_1+3]['upper_bound']=action_params["first_factor"]
                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_1+3]['iterator'])

                        #update the loop bounds if tiling is applied on loop 3
                        it_list[comp][depth_3]['upper_bound']=it_list[comp][depth_3]['upper_bound']/action_params["third_factor"]
                        it_list[comp][depth_3+2]={}
                        it_list[comp][depth_3+2]['iterator']="{}_1".format(it_list[comp][depth_3]['iterator'])   
                        it_list[comp][depth_3+2]['lower_bound']=it_list[comp][depth_3]['lower_bound']
                        it_list[comp][depth_3+2]['upper_bound']=action_params["third_factor"]
                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_3+2]['iterator'])
                    else:
                        if action_params["tiling_loop_1"]:
                            # print("in action params == 10 and tiling_loop_1")

                            it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                            it_list[comp][depth_1+3]={}
                            it_list[comp][depth_1+3]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])   
                            it_list[comp][depth_1+3]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                            it_list[comp][depth_1+3]['upper_bound']=action_params["first_factor"] 
                            #Add the new iterator to added_iterators
                            added_iterators.append(it_list[comp][depth_1+3]['iterator']) 

                        elif action_params["tiling_loop_2"]:
                            # print("in action params == 10 and tiling_loop_2")

                            it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                            it_list[comp][depth_2+2]={}
                            it_list[comp][depth_2+2]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                            it_list[comp][depth_2+2]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                            it_list[comp][depth_2+2]['upper_bound']=action_params["second_factor"]
                            #Add the new iterator to added_iterators
                            added_iterators.append(it_list[comp][depth_2+2]['iterator'])

                        elif action_params["tiling_loop_3"]:
                            # print("in action params == 10 and tiling_loop_3")

                            #update the loop bounds if tiling is applied on loop 1
                            it_list[comp][depth_3]['upper_bound']=it_list[comp][depth_3]['upper_bound']/action_params["third_factor"]
                            it_list[comp][depth_3+1]={}
                            it_list[comp][depth_3+1]['iterator']="{}_1".format(it_list[comp][depth_3]['iterator'])   
                            it_list[comp][depth_3+1]['lower_bound']=it_list[comp][depth_3]['lower_bound']
                            it_list[comp][depth_3+1]['upper_bound']=action_params["third_factor"]
                            #Add the new iterator to added_iterators
                            added_iterators.append(it_list[comp][depth_3+1]['iterator'])

            elif id in range(41,44): #Unrolling
                it_list[comp][action_params["dim_index"]]['upper_bound']=it_list[comp][action_params["dim_index"]]['upper_bound']/action_params['unrolling_factor']
            
            elif id in range(44,46):#Skewing
                depth_1=action_params["first_dim_index"]
                depth_2=action_params["second_dim_index"]

                l1_lower_bound=it_list[comp][depth_1]["lower_bound"]
                l1_upper_bound=it_list[comp][depth_1]["upper_bound"]
                l2_lower_bound=it_list[comp][depth_2]["lower_bound"]
                l2_upper_bound=it_list[comp][depth_2]["upper_bound"]

                l1_extent = abs(l1_upper_bound - l1_lower_bound)
                l2_extent = abs(l2_upper_bound - l2_lower_bound)

                l2_lower_bound = 0
                l1_lower_bound = abs(action_params["first_factor"]) * l1_lower_bound
                l1_upper_bound = l1_lower_bound + abs(action_params["first_factor"]) * l1_extent + abs(action_params["second_factor"]) * l2_extent
                l2_upper_bound = ((l1_extent * l2_extent) / (l1_upper_bound - l1_lower_bound)) + 1

                it_list[comp][depth_1]["lower_bound"]=l1_lower_bound
                it_list[comp][depth_1]["upper_bound"]=l1_upper_bound
                it_list[comp][depth_2]["lower_bound"]=l2_lower_bound
                it_list[comp][depth_2]["upper_bound"]=l2_upper_bound  
                
            elif id in range(48,56):#Reversal
                tmp=it_list[comp][action_params["dim_index"]]['lower_bound']
                it_list[comp][action_params["dim_index"]]['lower_bound']=it_list[comp][action_params["dim_index"]]['upper_bound']
                it_list[comp][action_params["dim_index"]]['upper_bound']=tmp 

        
        it_list=dict(sorted(it_list.items()))

        return it_list

    def apply_interchange(self, action_params):
        for comp in self.comps:
            l_code = "L" + self.it_dict[comp][action_params["first_dim_index"]]['iterator']
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Interchanged"]] = 1
            l_code = "L" + self.it_dict[comp][action_params["second_dim_index"]]['iterator']
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Interchanged"]] = 1

        iterators=list(self.annotations["iterators"].keys())
        if self.it_dict[comp][action_params["first_dim_index"]]['iterator'] in iterators:
            loop_1=iterators.index(self.it_dict[comp][action_params["first_dim_index"]]['iterator'])
        elif self.it_dict[comp][action_params["first_dim_index"]]['iterator'] in self.added_iterators:
            loop_1=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][action_params["first_dim_index"]]['iterator'])  
        self.obs["loops_representation"][loop_1][2]=1
        
        if self.it_dict[comp][action_params["second_dim_index"]]['iterator'] in iterators:
            loop_2=iterators.index(self.it_dict[comp][action_params["second_dim_index"]]['iterator'])
        elif self.it_dict[comp][action_params["second_dim_index"]]['iterator'] in self.added_iterators:
            loop_2=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][action_params["second_dim_index"]]['iterator'])  
        self.obs["loops_representation"][loop_2][2]=1

        for i in range(28):
            self.obs["action_mask"][i]=0
        for i in range(56,61):
            self.obs["action_mask"][i]=0
        
        for comp in self.comps:
            dim = self.schedule_dict[comp]["dim"]
            interchange_matrix = np.eye(dim,dim)
            first_iter_index = action_params["first_dim_index"]
            second_iter_index = action_params["second_dim_index"]
            interchange_matrix[first_iter_index, first_iter_index] = 0
            interchange_matrix[second_iter_index, second_iter_index] = 0
            interchange_matrix[first_iter_index, second_iter_index] = 1
            interchange_matrix[second_iter_index, first_iter_index] = 1
            self.schedule_dict[comp]["transformation_matrices"].append(interchange_matrix)
            self.schedule_dict[comp]["transformation_matrix"] =  interchange_matrix @ self.schedule_dict[comp]["transformation_matrix"]

    def apply_tiling(self, action_params):
        for comp in self.comps:
            comp_index=self.comp_indic_dict[comp]
       
            first_dim_index=action_params["first_dim_index"]
            second_dim_index=action_params["second_dim_index"]
            self.schedule_dict[comp]['tiling']= {'tiling_depth': action_params["tiling_depth"],
                                        'tiling_dims': [self.it_dict[comp][first_dim_index]['iterator'], self.it_dict[comp][second_dim_index]['iterator']],
                                        'tiling_factors': [action_params["first_factor"], action_params["second_factor"]]}
            l_code = "L" + self.it_dict[comp][first_dim_index]['iterator']
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Tiled"]] = 1
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "TileFactor"]] = action_params[
                "first_factor"
            ]

            #update the loop bounds if tiling is applied on loop 1
            if action_params["tiling_loop_1"]:
                # print("inside loop tiling 1")
                new_upper_bound_1=self.obs["representation"][self.comp_indic_dict[comp]][first_dim_index*20+1]/action_params["first_factor"]
                self.obs["representation"][self.comp_indic_dict[comp]][first_dim_index*20+1]=new_upper_bound_1
                new_inner_upper_bound_1=action_params["first_factor"]
                self.obs["representation"][self.comp_indic_dict[comp]][first_dim_index*20+10]=new_inner_upper_bound_1
                # print("after loop tiling 1")
                #Add the loop representation of the newly added iterator
                loop_added="{}_1".format(self.it_dict[comp][first_dim_index]['iterator'])
                self.added_iterators.append(loop_added)
                loop_index=len(self.annotations['iterators']) + self.added_iterators.index(loop_added)
                #Initialize lower and upper bounds
                loop_repr=[]
                if self.obs["representation"][comp_index][self.placeholders[comp][l_code + "Reversed"]]==1:
                    lower_bound=self.obs["representation"][comp_index][second_dim_index*20+1]
                else:
                    lower_bound=self.obs["representation"][comp_index][second_dim_index*20]                
                loop_repr.extend([lower_bound, action_params["first_factor"]])
                #Initialize the different tags
                loop_repr.extend([0,0,0,0,0,0,0,0,0,0,0])
                loop_log_rep = list(np.log1p(loop_repr))
                loop_repr.extend(loop_log_rep)
                self.obs["loops_representation"][loop_index]=loop_repr

            l_code = "L" + self.it_dict[comp][second_dim_index]['iterator']
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Tiled"]] = 1
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "TileFactor"]] = action_params[
                "second_factor"
            ]
            #update the loop bounds if tiling is applied on loop 2
            if action_params["tiling_loop_2"]:
                # print("inside loop tiling 2")
                new_upper_bound_2=self.obs["representation"][self.comp_indic_dict[comp]][second_dim_index*20+1]/action_params["second_factor"]
                self.obs["representation"][self.comp_indic_dict[comp]][second_dim_index*20+1]=new_upper_bound_2
                new_inner_upper_bound_2=action_params["second_factor"]
                self.obs["representation"][self.comp_indic_dict[comp]][second_dim_index*20+10]=new_inner_upper_bound_2
                # print("after loop tiling 2")

                #Add the loop representation of the newly added iterator
                loop_added="{}_1".format(self.it_dict[comp][second_dim_index]['iterator'])
                self.added_iterators.append(loop_added)
                loop_index=len(self.annotations['iterators']) + self.added_iterators.index(loop_added)
                #Initialize lower and upper bounds
                loop_repr=[]

                if self.obs["representation"][comp_index][self.placeholders[comp][l_code + "Reversed"]]==1:
                    lower_bound=self.obs["representation"][comp_index][second_dim_index*20+1]
                else:
                    lower_bound=self.obs["representation"][comp_index][second_dim_index*20]
                loop_repr.extend([lower_bound, action_params["second_factor"]])

                #Initialize the different tags
                loop_repr.extend([0,0,0,0,0,0,0,0,0,0,0])
                loop_log_rep = list(np.log1p(loop_repr))
                loop_repr.extend(loop_log_rep)
                self.obs["loops_representation"][loop_index]=loop_repr

            if action_params["tiling_depth"] == 3:
                third_dim_index=action_params["third_dim_index"]
                self.schedule_dict[comp]['tiling']= {'tiling_depth': action_params["tiling_depth"],
                                        'tiling_dims': [self.it_dict[comp][first_dim_index]['iterator'],
                                                        self.it_dict[comp][second_dim_index]['iterator'],
                                                        self.it_dict[comp][third_dim_index]['iterator']],
                                        'tiling_factors': [action_params["first_factor"],
                                                            action_params["second_factor"],
                                                            action_params["third_factor"]]}
                l_code = "L" + self.it_dict[comp][third_dim_index]['iterator']
                self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Tiled"]] = 1
                self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "TileFactor"]] = action_params[
                    "third_factor"
                ]
                #update the loop bounds if tiling is applied on loop 3
                if action_params["tiling_loop_3"]:
                    # print("inside loop tiling 3")
                    new_upper_bound_3=self.obs["representation"][self.comp_indic_dict[comp]][third_dim_index*20+1]/action_params["third_factor"]
                    self.obs["representation"][self.comp_indic_dict[comp]][third_dim_index*20+1]=new_upper_bound_3
                    new_inner_upper_bound_3=action_params["third_factor"]
                    self.obs["representation"][self.comp_indic_dict[comp]][third_dim_index*20+10]=new_inner_upper_bound_3
                    # print("after loop tiling 3")

                    #Add the loop representation of the newly added iterator
                    loop_added="{}_1".format(self.it_dict[comp][third_dim_index]['iterator'])
                    self.added_iterators.append(loop_added)
                    loop_index=len(self.annotations['iterators']) + self.added_iterators.index(loop_added)
                    #Initialize lower and upper bounds
                    loop_repr=[]
                    if self.obs["representation"][comp_index][self.placeholders[comp][l_code + "Reversed"]]==1:
                        lower_bound=self.obs["representation"][comp_index][third_dim_index*20+1]
                    else:
                        lower_bound=self.obs["representation"][comp_index][third_dim_index*20]

                    loop_repr.extend([lower_bound,action_params["third_factor"]])
                    #Initialize the different tags
                    loop_repr.extend([0,0,0,0,0,0,0,0,0,0,0])
                    loop_log_rep = list(np.log1p(loop_repr))
                    loop_repr.extend(loop_log_rep)
                    self.obs["loops_representation"][loop_index]=loop_repr

        #Update the loops representation
        iterators=list(self.annotations["iterators"].keys())

        if self.it_dict[comp][first_dim_index]['iterator'] in iterators:
            loop_1=iterators.index(self.it_dict[comp][first_dim_index]['iterator'])
        elif self.it_dict[comp][first_dim_index]['iterator'] in self.added_iterators:
            loop_1=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][first_dim_index]['iterator'])

        self.obs["loops_representation"][loop_1][3]=1
        self.obs["loops_representation"][loop_1][4]=action_params['first_factor']

        if self.it_dict[comp][second_dim_index]['iterator'] in iterators:
            loop_2=iterators.index(self.it_dict[comp][second_dim_index]['iterator'])
        elif self.it_dict[comp][second_dim_index]['iterator'] in self.added_iterators:
            loop_2=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][second_dim_index]['iterator'])  

        self.obs["loops_representation"][loop_2][3]=1
        self.obs["loops_representation"][loop_2][4]=action_params['second_factor']

        #Update the loop representation
        if action_params["tiling_depth"] == 3:

            if self.it_dict[comp][third_dim_index]['iterator'] in iterators:
                loop_3=iterators.index(self.it_dict[comp][third_dim_index]['iterator'])
            elif self.it_dict[comp][third_dim_index]['iterator'] in self.added_iterators:
                loop_3=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][third_dim_index]['iterator'])  

            self.obs["loops_representation"][loop_3][3]=1
            self.obs["loops_representation"][loop_3][4]=action_params['third_factor']
            
            
            if self.is_interchaged == False:

                if len(self.common_it) == 5:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.INTERCHANGE05, Action.INTERCHANGE06, Action.INTERCHANGE07, Action.INTERCHANGE15, Action.INTERCHANGE16, Action.INTERCHANGE17, 
                        Action.INTERCHANGE25, Action.INTERCHANGE26, Action.INTERCHANGE27, Action.INTERCHANGE35, Action.INTERCHANGE36, Action.INTERCHANGE37, 
                        Action.INTERCHANGE45, Action.INTERCHANGE46, Action.INTERCHANGE47,Action.INTERCHANGE56,Action.INTERCHANGE57, Action.INTERCHANGE67]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.INTERCHANGE05, Action.INTERCHANGE06, Action.INTERCHANGE15, Action.INTERCHANGE16, Action.INTERCHANGE25, Action.INTERCHANGE26, 
                        Action.INTERCHANGE35, Action.INTERCHANGE36, Action.INTERCHANGE45, Action.INTERCHANGE46, Action.INTERCHANGE56]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][[Action.INTERCHANGE05, Action.INTERCHANGE15, Action.INTERCHANGE25,Action.INTERCHANGE35, Action.INTERCHANGE45]]=1

                if len(self.common_it) == 4:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.INTERCHANGE04, Action.INTERCHANGE05, Action.INTERCHANGE06, Action.INTERCHANGE14, Action.INTERCHANGE15, Action.INTERCHANGE16, 
                        Action.INTERCHANGE24, Action.INTERCHANGE25, Action.INTERCHANGE26, Action.INTERCHANGE34, Action.INTERCHANGE35, Action.INTERCHANGE36, 
                        Action.INTERCHANGE45, Action.INTERCHANGE46, Action.INTERCHANGE56]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.INTERCHANGE04, Action.INTERCHANGE05, Action.INTERCHANGE14, Action.INTERCHANGE15,
                        Action.INTERCHANGE24, Action.INTERCHANGE25, Action.INTERCHANGE34, Action.INTERCHANGE35, Action.INTERCHANGE45]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][[Action.INTERCHANGE04, Action.INTERCHANGE14, Action.INTERCHANGE24, Action.INTERCHANGE34]]=1    

                if len(self.common_it) == 3:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.INTERCHANGE03, Action.INTERCHANGE04, Action.INTERCHANGE05, Action.INTERCHANGE13, Action.INTERCHANGE14, Action.INTERCHANGE15, 
                        Action.INTERCHANGE23, Action.INTERCHANGE24, Action.INTERCHANGE25, Action.INTERCHANGE34, Action.INTERCHANGE35, 
                        Action.INTERCHANGE45]]=1    
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.INTERCHANGE03, Action.INTERCHANGE04, Action.INTERCHANGE13, Action.INTERCHANGE14,
                        Action.INTERCHANGE23, Action.INTERCHANGE24, Action.INTERCHANGE34]]=1 
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][[Action.INTERCHANGE03, Action.INTERCHANGE13, Action.INTERCHANGE23]]=1 
                
                if len(self.common_it) == 2:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.INTERCHANGE02, Action.INTERCHANGE03, Action.INTERCHANGE04, Action.INTERCHANGE12, Action.INTERCHANGE13, Action.INTERCHANGE14, 
                        Action.INTERCHANGE23, Action.INTERCHANGE24, Action.INTERCHANGE34]]=1    
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.INTERCHANGE02, Action.INTERCHANGE03, Action.INTERCHANGE12, Action.INTERCHANGE13, Action.INTERCHANGE23]]=1 
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][[Action.INTERCHANGE02, Action.INTERCHANGE12]]=1 

                if len(self.common_it) == 1:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.INTERCHANGE01, Action.INTERCHANGE02, Action.INTERCHANGE03, Action.INTERCHANGE12, Action.INTERCHANGE13, Action.INTERCHANGE23]]=1    
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.INTERCHANGE01, Action.INTERCHANGE02, Action.INTERCHANGE12, Action.INTERCHANGE13]]=1    
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][[Action.INTERCHANGE01]]=1  

            if self.is_reversed == False:
                if len(self.common_it) == 5:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.REVERSAL5,Action.REVERSAL6, Action.REVERSAL7]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.REVERSAL5,Action.REVERSAL6]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][Action.REVERSAL5]=1

                elif len(self.common_it) == 4:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.REVERSAL4,Action.REVERSAL5, Action.REVERSAL6]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.REVERSAL4,Action.REVERSAL5]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][Action.REVERSAL4]=1

                elif len(self.common_it) == 3:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.REVERSAL3,Action.REVERSAL4, Action.REVERSAL5]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.REVERSAL3,Action.REVERSAL4]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][Action.REVERSAL3]=1

                elif len(self.common_it) == 2:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.REVERSAL2,Action.REVERSAL3, Action.REVERSAL4]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.REVERSAL2,Action.REVERSAL3]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][Action.REVERSAL2]=1

                elif len(self.common_it) == 1:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.REVERSAL1,Action.REVERSAL2, Action.REVERSAL3]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[Action.REVERSAL1,Action.REVERSAL2]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][Action.REVERSAL1]=1
        
        for i in range(28,41):
            self.obs["action_mask"][i]=0

        for i in range(56,61):
            self.obs["action_mask"][i]=0

    def apply_unrolling(self, action_params):

        for comp in self.comps:
            print(comp)
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp]["Unrolled"]] = 1
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp]["UnrollFactor"]] = action_params[comp]["unrolling_factor"]

            l_code = "L" + self.it_dict[comp][action_params[comp]["dim_index"]]['iterator']
            index_upper_bound=self.placeholders[comp][l_code+'Interchanged']-1
            self.obs["representation"][self.comp_indic_dict[comp]][index_upper_bound]=self.obs["representation"][self.comp_indic_dict[comp]][index_upper_bound]/action_params[comp]["unrolling_factor"]

            #Update the loop representation
            iterators=list(self.annotations["iterators"].keys())
            if self.it_dict[comp][action_params[comp]["dim_index"]]['iterator'] in iterators:
                loop_index=iterators.index(self.it_dict[comp][action_params[comp]["dim_index"]]['iterator'])
            elif self.it_dict[comp][action_params[comp]["dim_index"]]['iterator'] in self.added_iterators:
                loop_index=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][action_params[comp]["dim_index"]]['iterator'])           
            self.obs["loops_representation"][loop_index][5]=1
            self.obs["loops_representation"][loop_index][6]=action_params[comp]['unrolling_factor']

        for i in range(41,44):
            self.obs["action_mask"][i]=0
        for i in range(56,61):
            self.obs["action_mask"][i]=0
        
        print("1.6")
        try:
            for comp in self.comps:
                self.schedule_dict[comp]["unrolling_factor"] = action_params[comp]["unrolling_factor"]
        except Exception:
            print("ERROR_MODEL",traceback.format_exc())

        print("1.7")

    def apply_skewing(self, action_params):
        dim_1=action_params["first_dim_index"]
        dim_2=action_params["second_dim_index"]

        for comp in self.comps:
            l1_code = "L" + self.it_dict[comp][dim_1]['iterator']
            l2_code = "L" + self.it_dict[comp][dim_2]['iterator']

            #to get the start of the iterator in the representation template (just after the bounds)
            index1_upper_bound=self.placeholders[comp][l1_code+'Interchanged']-1
            index1_lower_bound=self.placeholders[comp][l1_code+'Interchanged']-2
            index2_upper_bound=self.placeholders[comp][l2_code+'Interchanged']-1
            index2_lower_bound=self.placeholders[comp][l2_code+'Interchanged']-2

            l1_lower_bound=self.obs["representation"][self.comp_indic_dict[comp]][index1_lower_bound]
            l1_upper_bound=self.obs["representation"][self.comp_indic_dict[comp]][index1_upper_bound]
            l2_lower_bound=self.obs["representation"][self.comp_indic_dict[comp]][index2_lower_bound]
            l2_upper_bound=self.obs["representation"][self.comp_indic_dict[comp]][index2_upper_bound]

            l1_extent = l1_upper_bound - l1_lower_bound
            l2_extent = l2_upper_bound - l2_lower_bound

            skew_factor = action_params["first_factor"]
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l1_code + "Skewed"]] = 1
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l1_code + "SkewFactor"]] = skew_factor
            self.obs["representation"][self.comp_indic_dict[comp]][index1_lower_bound]= abs(action_params["first_factor"]) * l1_lower_bound
            self.obs["representation"][self.comp_indic_dict[comp]][index1_upper_bound]= l1_lower_bound + abs(action_params["first_factor"]) * l1_extent + abs(action_params["second_factor"]) * l2_extent

            skew_factor = action_params["second_factor"]
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l2_code + "Skewed"]] = 1
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l2_code + "SkewFactor"]] = skew_factor
            self.obs["representation"][self.comp_indic_dict[comp]][index2_lower_bound]= 0
            self.obs["representation"][self.comp_indic_dict[comp]][index2_upper_bound]=(l2_extent) + 1

        #Update the loop representation
        iterators=list(self.annotations["iterators"].keys())
        if self.it_dict[comp][dim_1]['iterator'] in iterators:
            loop_1=iterators.index(self.it_dict[comp][dim_1]['iterator'])
        elif self.it_dict[comp][dim_1]['iterator'] in self.added_iterators:
            loop_1=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][dim_1]['iterator'])        
        self.obs["loops_representation"][loop_1][7]=1
        self.obs["loops_representation"][loop_1][8]=action_params['first_factor']
        #Skewing is applied on common loop levels so loop bounds are equal for all computations
        self.obs["loops_representation"][loop_1][9]=self.obs["representation"][0][index1_upper_bound]-self.obs["representation"][0][index1_lower_bound]

        if self.it_dict[comp][dim_2]['iterator'] in iterators:
            loop_2=iterators.index(self.it_dict[comp][dim_2]['iterator'])
        elif self.it_dict[comp][dim_2]['iterator'] in self.added_iterators:
            loop_2=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][dim_2]['iterator']) 
        self.obs["loops_representation"][loop_2][7]=1
        self.obs["loops_representation"][loop_2][8]=action_params['second_factor']
        self.obs["loops_representation"][loop_2][9]=self.obs["representation"][0][index2_upper_bound]-self.obs["representation"][0][index2_lower_bound]

        self.obs["action_mask"][44]=0
        self.obs["action_mask"][45]=0
        for i in range(56,61):
            self.obs["action_mask"][i]=0
        
        for comp in self.comps:
            dim = self.schedule_dict[comp]["dim"]
            skewing_matrix = np.eye(dim,dim)
            first_iter_index = action_params["first_dim_index"]
            second_iter_index = action_params["second_dim_index"]
            first_factor = action_params["first_factor"]
            second_factor = action_params["second_factor"]
            if (first_factor, second_factor) in global_dioph_sols_dict:
                a, b = global_dioph_sols_dict[(first_factor, second_factor)]
            else:
                a, b = linear_diophantine_default(first_factor, second_factor)

            skewing_matrix[first_iter_index, first_iter_index] = first_factor
            skewing_matrix[first_iter_index, second_iter_index] = second_factor
            skewing_matrix[second_iter_index, first_iter_index] = a
            skewing_matrix[second_iter_index, second_iter_index] = b
            self.schedule_dict[comp]["transformation_matrices"].append(skewing_matrix)
            self.schedule_dict[comp]["transformation_matrix"] = skewing_matrix @ self.schedule_dict[comp]["transformation_matrix"]

    def apply_parallelization(self, action_params):
        first_comp=list(self.it_dict.keys())[0]
        iterator = self.it_dict[first_comp][action_params["dim_index"]]['iterator']
        self.schedule_dict[first_comp]["parallelized_dim"] = iterator
        l_code = "L" + iterator

        self.obs["representation"][0][self.placeholders[first_comp][l_code + "Parallelized"]] = 1

        #Update the loop representation
        iterators=list(self.annotations["iterators"].keys())
        if self.it_dict[first_comp][action_params["dim_index"]]['iterator'] in iterators:
            loop_index=iterators.index(self.it_dict[first_comp][action_params["dim_index"]]['iterator'])
        elif self.it_dict[first_comp][action_params["dim_index"]]['iterator'] in self.added_iterators:
            loop_index=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[first_comp][action_params["dim_index"]]['iterator'])
        self.obs["loops_representation"][loop_index][10]=1
        #Update the action mask
        self.obs["action_mask"][46]=0
        self.obs["action_mask"][47]=0
        for i in range(56,61):
            self.obs["action_mask"][i]=0
        print("The first comp is ", first_comp)
        print("The result is ", self.schedule_dict[first_comp]["parallelized_dim"])

    def apply_reversal(self, action_params):
        for comp in self.comps:
            l_code = "L" + self.it_dict[comp][action_params["dim_index"]]['iterator']

            index_upper_bound=self.placeholders[comp][l_code+'Interchanged']-1
            index_lower_bound=self.placeholders[comp][l_code+'Interchanged']-2

            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Reversed"]] = 1

            tmp=self.obs["representation"][self.comp_indic_dict[comp]][index_lower_bound]
            self.obs["representation"][self.comp_indic_dict[comp]][index_lower_bound]=self.obs["representation"][self.comp_indic_dict[comp]][index_upper_bound]
            self.obs["representation"][self.comp_indic_dict[comp]][index_upper_bound]=tmp 

        #Update the loop representation
        iterators=list(self.annotations["iterators"].keys())
        if self.it_dict[comp][action_params["dim_index"]]['iterator'] in iterators:
            loop_index=iterators.index(self.it_dict[comp][action_params["dim_index"]]['iterator'])
        elif self.it_dict[comp][action_params["dim_index"]]['iterator'] in self.added_iterators:
            loop_index=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][action_params["dim_index"]]['iterator'])        
        self.obs["loops_representation"][loop_index][11]=1

        for i in range(48,56):
            self.obs["action_mask"][i]=0
        for i in range(56,61):
            self.obs["action_mask"][i]=0
        
        for comp in self.comps:
            dim = self.schedule_dict[comp]["dim"]
            reversal_matrix = np.eye(dim,dim)
            dim_index = action_params["dim_index"]
            reversal_matrix[dim_index, dim_index] = -1
            self.schedule_dict[comp]["transformation_matrices"].append(reversal_matrix)
            self.schedule_dict[comp]["transformation_matrix"] = reversal_matrix @ self.schedule_dict[comp]["transformation_matrix"]
    
    def apply_fusion(self, action_params):
        fusion = []
        for comp in action_params["fuse_comps"]:
            fusion.append(comp)
            l_code = "L" + self.it_dict[comp][action_params["dim_index"]]['iterator']
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Fused"]] = 1
        fusion.append(action_params["dim_index"])
        self.schedule_dict["fusions"].append(fusion)
        #Update the loop representation
        iterators=list(self.annotations["iterators"].keys())
        if self.it_dict[comp][action_params["dim_index"]]['iterator'] in iterators:
            loop_index=iterators.index(self.it_dict[comp][action_params["dim_index"]]['iterator'])
        elif self.it_dict[comp][action_params["dim_index"]]['iterator'] in self.added_iterators:
            loop_index=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][action_params["dim_index"]]['iterator'])        
        self.obs["loops_representation"][loop_index][12]=1

        for i in range(56,61):
            self.obs["action_mask"][i]=0


    def get_exec_time_by_model(self,optims_list, cmd_type, nb_executions, initial_exec_time):
        self.schedule_list_model.append({
            "schedule_str":self.schedule_str,
            "schedule_dict":self.schedule_dict
        })
        print(f"schedule={self.schedule_str};",end="")
        stat=dict()
        try:
            print("Done saving")
            print("Done saving")
            computations_tensor, loops_tensor = self.get_schedule_representation(self.annotations,
                                        self.schedule_dict,
                                        self.templates["comps_repr_templates_list"],
                                        self.templates["loops_repr_templates_list"],
                                        self.templates["comps_placeholders_indices_dict"],
                                        self.templates["loops_placeholders_indices_dict"],
                                        max_depth = self.MAX_DEPTH-1)
            # print(computations_tensor.shape, loops_tensor.shape)
            tree_tensors = (self.templates["prog_tree"], computations_tensor, loops_tensor)
            with torch.no_grad():
                predicted_speedup = self.model(tree_tensors,num_matrices=self.MAX_DEPTH-1).item()
                stat["initial_execution_time"]=self.prog.initial_execution_time
                # print("initial_execution_time", self.prog.initial_execution_time)
                stat["predicted_speedup"]=predicted_speedup
                print(f"predicted_speedup={predicted_speedup}")
                stat["predicted_execution_time"]=self.prog.initial_execution_time/predicted_speedup
                # print("predicted_execution_time", self.prog.initial_execution_time/predicted_speedup)
        except Exception:
            print("ERROR_MODEL",traceback.format_exc())
            # or
            print(sys.exc_info()[2])

        return  stat["predicted_execution_time"]

    def get_exec_time(self):

        # print("in get_exec_time")l

        prog_name= self.prog.name
        execution_time=0
        if self.schedule_str != "" and self.schedule != []:
            if prog_name in self.scheds.keys():
                #print("Am in 1")

                if self.schedule_str in self.scheds[prog_name]:
                    #print("Am in 1.1")
                    # print("Prog in sched: True, sched in scheds: True")
                    execution_time=self.scheds[prog_name][self.schedule_str][0]
                    # print("**out of ** Prog in sched: True, sched in scheds: False")

                else:  
                    #print("Am in 1.2")
                    
                    if prog_name in self.new_scheds.keys() and self.schedule_str in self.new_scheds[prog_name].keys():
                        #print("Am in 1.2.1")
                        # print("Prog in sched: True, sched in scheds: False, shced in new_scheds: True")
                        execution_time=self.new_scheds[prog_name][self.schedule_str][1]
                        # print("**out of **Prog in sched: True, sched in scheds: False, shced in new_scheds: True")
                    else:
                        ## print("Am in 1.2.2")
                        curr_sched=copy.deepcopy(self.schedule)
                        # print("Prog in sched: True, sched in scheds: False, shced in new_scheds: False")
                        self.new_scheds[prog_name]={}
                        execution_time=self.measurement_env(self.schedule,'sched_eval',self.nb_executions, self.prog.initial_execution_time)
                        self.new_scheds[prog_name][self.schedule_str]=(curr_sched,execution_time,0)
                        # print("**out of **Prog in sched: True, sched in scheds: False, shced in new_scheds: False")

                    
            else:

                ## print("Am in 2")
                if prog_name in self.new_scheds.keys():
                    ## print("Am in 2.1")

                    if self.schedule_str in self.new_scheds[prog_name].keys():
                        ## print("Am in 2.1.1")
                        # print("Prog in sched: False, sched in scheds: False Prog in new sched: True, sched in new scheds: True")
                        execution_time=self.new_scheds[prog_name][self.schedule_str][1]
                        # print("** out of** Prog in sched: False, sched in scheds: False Prog in new sched: True, sched in new scheds: True")
                        

                    else:
                        ## print("Am in 2.1.2")
                        curr_sched=copy.deepcopy(self.schedule)
                        # print("Prog in sched: False, sched in scheds: False Prog in new sched: True, sched in new scheds: False")
                        execution_time=self.measurement_env(self.schedule,'sched_eval',self.nb_executions, self.prog.initial_execution_time)
                        self.new_scheds[prog_name][self.schedule_str]=(curr_sched,execution_time,0)
                        # print("** out of** Prog in sched: False, sched in scheds: False Prog in new sched: True, sched in new scheds: False")
                        

                else:
                    ## print("Am in 2.2")
                    curr_sched=copy.deepcopy(self.schedule)
                    # print("Prog in sched: False, sched in scheds: False Prog in new sched: False")
                    self.new_scheds[prog_name]={}
                    start_time=time.time()
                    execution_time=self.measurement_env(self.schedule,'sched_eval',self.nb_executions, self.prog.initial_execution_time)
                    sched_time=time.time()-start_time
                    self.codegen_total_time+=sched_time

                    self.new_scheds[prog_name][self.schedule_str]=(curr_sched,execution_time,0)
                    # print("**out of **Prog in sched: True, sched in scheds: False, shced in new_scheds: False")

        else:
            execution_time=self.prog.initial_execution_time
                    
        # print("get_exec_time returned {} for the function {}".format(execution_time,self.prog.name))
        return execution_time