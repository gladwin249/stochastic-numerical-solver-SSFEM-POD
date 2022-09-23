"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NAME OF FILE: SSFEM_3D_wgd_iris_lossless_four_circular_diel_post_epsr_24_ROM_freq_sweep_4_github
%
% PURPOSE: Stochastic modeling using SSFEM-POD for permittivity
%          variations in dielectric loaded waveguide
%
% Written by Gladwin Jos
%
% Date : 19/01/2021  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


# Reference:
# An Efficient  SSFEM-POD scheme for Wideband Stochastic Analysis of Permittivity Variations  
import matplotlib.pyplot as plt 
import parse_gmsh_3D as pg3
import numpy as np
import edge_connect_modified_3D as ecm3
import find_edges_3D as fed3
import math
import random_number_seed as rd 
import cmath 
import time 
import det_fem_wgd_3D_sparse as dfw3s
import wgd_3D_boundary_matrix_sparse as w3bms
import wgd_3D_boundary_matrix as w3bm 
import latin_hypercube_sampling as lhs
import orthogonal_polynomial as op
from scipy.sparse.linalg import spsolve
from scipy import sparse 
import Psi_j_property as pj 
import generate_matrices_3D_wgd_SSFEM_lossless_four_diel_rom_sparse as gm3sl4rs 
import generate_boundary_excitation_matrices_3D_wgd_hdm as gbe3w
import generate_boundary_excitation_matrices_3D_wgd_hdm_sparse as gbe3ws
import gamma_trans_evaluation as gte
import dask.array as da # for svd of large thin matrices

plt.close('all') # close all figures 
time_start=time.clock() 
N_SSFEM=10000 
deviation_percent=10 # change accordingly
No_of_dielectric_layer=4 
file_name="3d_wgd_four_lossless_circular_dielectric_mesh_finer_refined_corner_iris_24_Ne_24613.msh" 
# Mesh statistics
#Ne=24613,D1=1970,D2=1703,D3=1761, D4=1869
node_list,three_node_triangle, \
    four_node_tetrahedron=pg3.parse_gmsh_3D(file_name)
Ne=len(four_node_tetrahedron) 
n=np.zeros((Ne,4))
for ii in range(Ne):
    n[ii][0]=four_node_tetrahedron[ii][0]-1 # since the index
    n[ii][1]=four_node_tetrahedron[ii][1]-1 # starts with 0
    n[ii][2]=four_node_tetrahedron[ii][2]-1
    n[ii][3]=four_node_tetrahedron[ii][3]-1
x=node_list[:,0]
y=node_list[:,1] 
z=node_list[:,2]
x=np.round(x,6)
y=np.round(y,6)
z=np.round(z,6)
x_max=max(x) 
x_min=min(x) 
y_max=max(y) 
y_min=min(y) 
z_min=min(z) 
z_max=max(z) 
if z_max == 0: 
    temp=z_max 
    z_max=z_min 
    z_min=temp  
input_len=z_min 
output_len=z_max 
No_of_nodes=len(x)   
tet_edge=np.array([[0, 1],[0, 2],[0, 3],[1, 2],[3, 1],[2, 3]])
local_edge= ecm3.edge_array_local_3D(n,Ne,tet_edge)      
global_edge,local_edge_array_elm_no=ecm3.edge_array_global_3D_unique(local_edge) 
No_of_edges=len(global_edge)  
edge_element=ecm3.element_edge_array_3D_reshape(Ne,local_edge_array_elm_no) 
sign_edge_element=ecm3.obtain_sign_edge_element(four_node_tetrahedron,Ne)
   # ( used four_node_tetrahedron instead of n inorder to prevent -0,+0 conflict)
local_face_array,face_no_local=ecm3.face_array_local_3D(n,Ne)
global_face_array,local_face_array_elm_no,global_face_array_element_no=\
        ecm3.face_array_global_3D_unique( local_face_array,face_no_local)
face_element_array=ecm3.element_face_array_3D_reshape(Ne,local_face_array_elm_no)
face_surface,face_surface_element= \
        ecm3.surface_faces_3D(global_face_array,face_element_array,\
            global_face_array_element_no,Ne) 
edge_surface=ecm3.surface_edges_3D_unique(face_surface)
edge_surface_element=ecm3.edge_surface_element(edge_surface,global_edge)
input_face_surface,input_face_surface_element_no,\
        output_face_surface,output_face_surface_element_no,pec_surface= \
        ecm3.input_output_face_surface(face_surface,node_list,face_surface_element,\
            input_len,output_len) 
input_normal_vector= \
        ecm3.find_normal_vector(input_face_surface,node_list,\
            input_face_surface_element_no,n,sign_edge_element) 
output_normal_vector= \
        ecm3.find_normal_vector(output_face_surface,node_list,\
            output_face_surface_element_no,n,sign_edge_element) 
pec_edge_surface,pec_edge_element_no=fed3.get_pec_edges(pec_surface,global_edge) 
edge_element_no_input=\
        fed3.get_input_output_edge_element(input_face_surface,global_edge) 
edge_element_no_output=\
        fed3.get_input_output_edge_element(output_face_surface,global_edge)
  
epsilon_r_1_dash=24 
epsilon_r_2_dash=24
epsilon_r_3_dash=24
epsilon_r_4_dash=24  
count_dielectric_1=0
count_dielectric_2=0
count_dielectric_3=0
count_dielectric_4=0
i=complex(0,1)

# defining dielectric dimension
err_value_1=0.00000001
err_value_2=0.000001  
# Dimension
D_1=0.01337 # changed as per Esteban's suggestion (2nd author)
D_2=0.00698 
D_3=0.006286 
D_4=0.00828 
D_5=0.0061 
D_6=0.01905 
D_7=D_6/2 
l=0.002   
x_c1=D_2+l+D_2/2
x_c=D_7
x_c2=D_2 +D_2+2*l+D_4/2
x_c3=D_2 +D_2+D_4+3*l+D_4/2
x_c4=D_2 +D_2+2*D_4+4*l+D_2/2
r_1=0.002111
r_2=0.002172
r_3=0.002172
r_4=0.002111
No_of_rv=No_of_dielectric_layer
xi_value=np.zeros((N_SSFEM,No_of_rv)) 
for ii in range(No_of_rv):
    xi_value[:,ii]=\
            rd.random_number_seed_gaussian(ii+1,N_SSFEM)
i=complex(0,1)
epsilon_r_wgd_det=np.zeros(Ne,dtype=complex) 
for e in range(Ne): 
    node_1=int(n[e][0])
    node_2=int(n[e][1])
    node_3=int(n[e][2])
    node_4=int(n[e][3])
    if  ((((z[node_1]-x_c1)**2+(x[node_1]-x_c)**2<=(r_1**2+err_value_1))& ((y_min<=y[node_1])&(y[node_1]<=y_max))) &\
           (((z[node_2]-x_c1)**2+(x[node_2]-x_c)**2<=(r_1**2+err_value_1))& ((y_min<=y[node_2])&(y[node_2]<=y_max))) & \
         (((z[node_3]-x_c1)**2+(x[node_3]-x_c)**2<=(r_1**2+err_value_1))& ((y_min<=y[node_3])&(y[node_3]<=y_max))) & \
         (((z[node_4]-x_c1)**2+(x[node_4]-x_c)**2<=(r_1**2+err_value_1))& ((y_min<=y[node_4])&(y[node_4]<=y_max))) ):
          count_dielectric_1=count_dielectric_1+1  
          epsilon_r_wgd_det[e]=epsilon_r_1_dash 
    elif ((((z[node_1]-x_c2)**2+(x[node_1]-x_c)**2<=(r_2**2+err_value_2))& ((y_min<=y[node_1])&(y[node_1]<=y_max))) &\
           (((z[node_2]-x_c2)**2+(x[node_2]-x_c)**2<=(r_2**2+err_value_2))& ((y_min<=y[node_2])&(y[node_2]<=y_max))) & \
         (((z[node_3]-x_c2)**2+(x[node_3]-x_c)**2<=(r_2**2+err_value_2))& ((y_min<=y[node_3])&(y[node_3]<=y_max))) & \
         (((z[node_4]-x_c2)**2+(x[node_4]-x_c)**2<=(r_2**2+err_value_2))& ((y_min<=y[node_4])&(y[node_4]<=y_max))) ):
          count_dielectric_2=count_dielectric_2+1  
          epsilon_r_wgd_det[e]=epsilon_r_2_dash 
    elif ((((z[node_1]-x_c3)**2+(x[node_1]-x_c)**2<=(r_3**2+err_value_2)) & ((y_min<=y[node_1])&(y[node_1]<=y_max)))&\
           (((z[node_2]-x_c3)**2+(x[node_2]-x_c)**2<=(r_3**2+err_value_2))& ((y_min<=y[node_2])&(y[node_2]<=y_max))) & \
        ( ((z[node_3]-x_c3)**2+(x[node_3]-x_c)**2<=(r_3**2+err_value_2))& ((y_min<=y[node_3])&(y[node_3]<=y_max))) & \
         (((z[node_4]-x_c3)**2+(x[node_4]-x_c)**2<=(r_3**2+err_value_2))& ((y_min<=y[node_4])&(y[node_4]<=y_max))) ):
          count_dielectric_3=count_dielectric_3+1  
          epsilon_r_wgd_det[e]=epsilon_r_3_dash 
    elif ((((z[node_1]-x_c4)**2+(x[node_1]-x_c)**2<=(r_4**2+err_value_1))& ((y_min<=y[node_1])&(y[node_1]<=y_max))) &\
          ( ((z[node_2]-x_c4)**2+(x[node_2]-x_c)**2<=(r_4**2+err_value_1))& ((y_min<=y[node_2])&(y[node_2]<=y_max))) & \
         (((z[node_3]-x_c4)**2+(x[node_3]-x_c)**2<=(r_4**2+err_value_1))& ((y_min<=y[node_3])&(y[node_3]<=y_max))) & \
         (((z[node_4]-x_c4)**2+(x[node_4]-x_c)**2<=(r_4**2+err_value_1)) & ((y_min<=y[node_4])&(y[node_4]<=y_max)))):
          count_dielectric_4=count_dielectric_4+1  
          epsilon_r_wgd_det[e]=epsilon_r_4_dash 
    else:
        epsilon_r_wgd_det[e]=1+0*i 
print(count_dielectric_1)
# dimensions
a=x_max-x_min 
b=y_max-y_min 
E0=1 
width=a 
height=b  
# constants
mu_r=1
mu_0=4*math.pi*10**(-7) 
epsilon_0=8.854187817*10**(-12)           
no_of_values_lhs=50 
xi_value_lhs=lhs.LHS_samples_gaussian(xi_value,No_of_rv,no_of_values_lhs)
B1ij_det_sparse,B2ij_det_sparse=w3bms.obtain_boundary_matrix_input_output_port_sparse(n,node_list,edge_element, \
    sign_edge_element,input_face_surface_element_no,output_face_surface_element_no,\
    No_of_edges,output_face_surface,input_face_surface,input_normal_vector,output_normal_vector)

no_of_divisions_initial=no_of_values_lhs 
freq_min=10.6*(10**9)   
freq_max=11.65*(10**9) 
freq_initial=np.linspace(freq_min,freq_max,no_of_divisions_initial)    
if deviation_percent==5: 
    sigma_i=np.array([0.05,0.05,0.05,0.05]) 
if deviation_percent==10:
    sigma_i=np.array([0.01,0.01,0.01,0.01])*deviation_percent 
         
U_matrix=np.zeros((No_of_edges-len(pec_edge_element_no),no_of_values_lhs),dtype=complex)
for kk in range(no_of_values_lhs):
    print(kk)
    k0=2*math.pi*freq_initial[kk]*math.sqrt(mu_0*epsilon_0) 
    if k0**2 >=(math.pi/a)**2:
        k_z_10=math.sqrt(k0**2-(math.pi/a)**2) 
    else:
        k_z_10=math.sqrt((math.pi/a)**2-k0**2)  
    gamma_input=k_z_10*i
    epsilon_r_wgd_stoc=np.zeros(Ne,dtype=complex) 
    lhs_samples_itr=xi_value_lhs[kk,:]
    for e in range(Ne): 
        node_1=int(n[e][0])
        node_2=int(n[e][1])
        node_3=int(n[e][2])
        node_4=int(n[e][3])
        if  ((((z[node_1]-x_c1)**2+(x[node_1]-x_c)**2<=(r_1**2+err_value_1))& ((y_min<=y[node_1])&(y[node_1]<=y_max))) &\
               (((z[node_2]-x_c1)**2+(x[node_2]-x_c)**2<=(r_1**2+err_value_1))& ((y_min<=y[node_2])&(y[node_2]<=y_max))) & \
             (((z[node_3]-x_c1)**2+(x[node_3]-x_c)**2<=(r_1**2+err_value_1))& ((y_min<=y[node_3])&(y[node_3]<=y_max))) & \
             (((z[node_4]-x_c1)**2+(x[node_4]-x_c)**2<=(r_1**2+err_value_1))& ((y_min<=y[node_4])&(y[node_4]<=y_max))) ):  
            epsilon_r_wgd_stoc[e]=epsilon_r_1_dash+sigma_i[0]*lhs_samples_itr[0]  
        elif ((((z[node_1]-x_c2)**2+(x[node_1]-x_c)**2<=(r_2**2+err_value_2))& ((y_min<=y[node_1])&(y[node_1]<=y_max))) &\
               (((z[node_2]-x_c2)**2+(x[node_2]-x_c)**2<=(r_2**2+err_value_2))& ((y_min<=y[node_2])&(y[node_2]<=y_max))) & \
             (((z[node_3]-x_c2)**2+(x[node_3]-x_c)**2<=(r_2**2+err_value_2))& ((y_min<=y[node_3])&(y[node_3]<=y_max))) & \
             (((z[node_4]-x_c2)**2+(x[node_4]-x_c)**2<=(r_2**2+err_value_2))& ((y_min<=y[node_4])&(y[node_4]<=y_max))) ):
            epsilon_r_wgd_stoc[e]=epsilon_r_2_dash+sigma_i[1]*lhs_samples_itr[1] 
        elif ((((z[node_1]-x_c3)**2+(x[node_1]-x_c)**2<=(r_3**2+err_value_2)) & ((y_min<=y[node_1])&(y[node_1]<=y_max)))&\
               (((z[node_2]-x_c3)**2+(x[node_2]-x_c)**2<=(r_3**2+err_value_2))& ((y_min<=y[node_2])&(y[node_2]<=y_max))) & \
            ( ((z[node_3]-x_c3)**2+(x[node_3]-x_c)**2<=(r_3**2+err_value_2))& ((y_min<=y[node_3])&(y[node_3]<=y_max))) & \
             (((z[node_4]-x_c3)**2+(x[node_4]-x_c)**2<=(r_3**2+err_value_2))& ((y_min<=y[node_4])&(y[node_4]<=y_max))) ):
            epsilon_r_wgd_stoc[e]=epsilon_r_3_dash+sigma_i[2]*lhs_samples_itr[2] 
        elif ((((z[node_1]-x_c4)**2+(x[node_1]-x_c)**2<=(r_4**2+err_value_1))& ((y_min<=y[node_1])&(y[node_1]<=y_max))) &\
              ( ((z[node_2]-x_c4)**2+(x[node_2]-x_c)**2<=(r_4**2+err_value_1))& ((y_min<=y[node_2])&(y[node_2]<=y_max))) & \
             (((z[node_3]-x_c4)**2+(x[node_3]-x_c)**2<=(r_4**2+err_value_1))& ((y_min<=y[node_3])&(y[node_3]<=y_max))) & \
             (((z[node_4]-x_c4)**2+(x[node_4]-x_c)**2<=(r_4**2+err_value_1)) & ((y_min<=y[node_4])&(y[node_4]<=y_max)))):
            epsilon_r_wgd_stoc[e]=epsilon_r_4_dash+sigma_i[3]*lhs_samples_itr[3] 
        else:
            epsilon_r_wgd_stoc[e]=1+0*i  
    EEij_det_sparse,FFij_det_sparse=dfw3s.obtain_E_and_F_deterministic_matrix_3D_waveguide_sparse \
        (mu_r,tet_edge,No_of_edges,n,node_list,epsilon_r_wgd_stoc,edge_element,sign_edge_element)  
    Kij_sparse=EEij_det_sparse-((k0**2)*FFij_det_sparse)+gamma_input*(B1ij_det_sparse+B2ij_det_sparse)  
    ffij=w3bm.obtain_excitation_matrix(No_of_edges,Ne,input_face_surface,edge_element, \
            input_face_surface_element_no,n,sign_edge_element, \
            input_normal_vector,node_list,k_z_10,E0,width)
    # applying boundary conditions
    # Deleting corresponding row of Kij of lil_matrix 
    Kij_sparse_csr=Kij_sparse.tocsr() #converting to csr matrix
    mask_rows=np.ones(Kij_sparse_csr.shape[0],dtype=bool)
    mask_columns=np.ones(Kij_sparse_csr.shape[1],dtype=bool) 
    mask_rows[abs(pec_edge_element_no)]=False 
    mask_columns[abs(pec_edge_element_no)]=False
    Kij_sparse_csr=Kij_sparse_csr[mask_rows]
    # Deleting corresponding column of Kij of lil_matrix 
    Kij_sparse_csr=Kij_sparse_csr[:,mask_columns]
    ffij=np.delete(ffij,abs(pec_edge_element_no),0)   
    E_field_samples=spsolve(Kij_sparse_csr,ffij)
    U_matrix[:,kk]=E_field_samples 
    del EEij_det_sparse
    del FFij_det_sparse
    del ffij
    del Kij_sparse
    del Kij_sparse_csr
    del E_field_samples 
   
   
# works only for small size matrix svd computations
# SVD_U,SVD_Sigma,SVD_V_T= np.linalg.svd(U_matrix)
U_matrix_dask=da.asarray(U_matrix) # converting into dask array
SVD_U_dask,SVD_Sigma_dask,SVD_V_T_dask= da.linalg.svd(U_matrix_dask)
SVD_Sigma=SVD_Sigma_dask.compute()
SVD_U=SVD_U_dask.compute()
SVD_V_T=SVD_V_T_dask.compute()
SVD_Sigma_temp=np.zeros((len(U_matrix),len(U_matrix[0])))
for ii in range(len(U_matrix[0])):
    SVD_Sigma_temp[ii][ii]=SVD_Sigma[ii]
#U_matrix_temp=SVD_U.dot(SVD_Sigma_temp).dot(SVD_V_T)
sing_values=SVD_Sigma
N_value=len(sing_values) 
RIC_value=np.zeros(N_value,dtype=float)
RIC_value_temp=np.zeros(N_value,dtype=float)
for ii in range(N_value):
    index_temp=np.linspace(0,ii,ii+1)
    index=np.zeros(ii+1,dtype=int)
    for jj in range(ii+1):
        index[jj]=int(index_temp[jj])
    RIC_value[ii]=sum(sing_values[index]*sing_values[index])/sum(sing_values*sing_values) 
            
err_value_arr=np.array([0.99999999,0.9999999,0.9999990,0.9999900,0.9999000,0.99900,0.99000\
                    ,0.90000,0.8,0.7])
m_count_list=np.zeros(len(err_value_arr))
for kk in range(len(err_value_arr)):
    break_loop=0 
    for ii in range(N_value):
        if (RIC_value[ii] >=err_value_arr[kk]) & ( break_loop==0):
            m_count_list[kk]=ii+1 # index starts with 0
            break_loop=1 
m_count=int(m_count_list[0])
#m_count=no_of_values_lhs
col_index=np.arange(0,m_count)
col_index_int=np.zeros(len(col_index),dtype=int)
for ii in range(len(col_index)):
    col_index_int[ii]= int(col_index[ii])
U_POD=SVD_U[:,col_index_int]

plt.figure(1)  
x_temp=np.linspace(1,len(sing_values),len(sing_values)) 
plt.bar(x_temp,sing_values)  
for ii in range(len(sing_values)):
    legend_value=plt.plot(x_temp[ii],sing_values[ii],'ok',markersize=3  ) 
plt.legend(legend_value,['Singular values'], loc='best') 

plt.xlabel('index, $i$') 
plt.ylabel('Singular values, $\sigma_1$')

print('Evaluated singular values')
# constants
mu_r=1
mu_0=4*math.pi*10**(-7) 
epsilon_0=8.854187817*10**(-12)  
no_of_divisions=100
freq_min=10.6*(10**9)   
freq_max=11.65*(10**9) 
freq_SSFEM=np.linspace(freq_min,freq_max,no_of_divisions)

# finding K_bar
K_mean,M_mean,K_i_real_mean =gm3sl4rs.generate_matrices_3D_wgd_SSFEM_lossless_four_circular_diel_rom_sparse(mu_r,tet_edge,\
        No_of_edges,n,node_list,epsilon_r_wgd_det,edge_element,sign_edge_element,\
        No_of_dielectric_layer,x_c1,x_c,x_c2,x_c3,x_c4,r_1,r_2,r_3,r_4,y_min,y_max,\
        err_value_1,err_value_2,sigma_i)   
B1_mean,B2_mean= gbe3ws.generate_boundary_matrices_3D_wgd_hdm_sparse(n,node_list,edge_element, \
    sign_edge_element,input_face_surface_element_no,output_face_surface_element_no,\
    No_of_edges,output_face_surface,input_face_surface,input_normal_vector,output_normal_vector)
U_incident_mean=lambda x_1:np.array([0,-2*i*E0*math.sin(math.pi*x_1/width),0])
f_mean=gbe3w.generate_excitation_matrix_3d_wgd_hdm(No_of_edges,Ne,input_face_surface,edge_element, \
        input_face_surface_element_no,n,sign_edge_element, \
        input_normal_vector,node_list,U_incident_mean,E0,width)
 
# applying PEC boundary conditions
# Deleting corresponding row of Kij of lil_matrix 
K_mean_sparse_csr=K_mean.tocsr() #converting to csr matrix
M_mean_sparse_csr=M_mean.tocsr() 
B1_mean_sparse_csr=B1_mean.tocsr() 
B2_mean_sparse_csr=B2_mean.tocsr() 
mask_rows=np.ones(K_mean_sparse_csr.shape[0],dtype=bool)
mask_columns=np.ones(K_mean_sparse_csr.shape[1],dtype=bool) 
mask_rows[abs(pec_edge_element_no)]=False 
mask_columns[abs(pec_edge_element_no)]=False
K_mean_sparse_csr=K_mean_sparse_csr[mask_rows]
K_mean_sparse_csr=K_mean_sparse_csr[:,mask_columns]
mask_rows=np.ones(M_mean_sparse_csr.shape[0],dtype=bool)
mask_columns=np.ones(M_mean_sparse_csr.shape[1],dtype=bool) 
mask_rows[abs(pec_edge_element_no)]=False 
mask_columns[abs(pec_edge_element_no)]=False
M_mean_sparse_csr=M_mean_sparse_csr[mask_rows]
M_mean_sparse_csr=M_mean_sparse_csr[:,mask_columns]
mask_rows=np.ones(B1_mean_sparse_csr.shape[0],dtype=bool)
mask_columns=np.ones(B1_mean_sparse_csr.shape[1],dtype=bool) 
mask_rows[abs(pec_edge_element_no)]=False 
mask_columns[abs(pec_edge_element_no)]=False
B1_mean_sparse_csr=B1_mean_sparse_csr[mask_rows]
B1_mean_sparse_csr=B1_mean_sparse_csr[:,mask_columns]
mask_rows=np.ones(B2_mean_sparse_csr.shape[0],dtype=bool)
mask_columns=np.ones(B2_mean_sparse_csr.shape[1],dtype=bool) 
mask_rows[abs(pec_edge_element_no)]=False 
mask_columns[abs(pec_edge_element_no)]=False
B2_mean_sparse_csr=B2_mean_sparse_csr[mask_rows]
# Deleting corresponding column of Kij of lil_matrix 
B2_mean_sparse_csr=B2_mean_sparse_csr[:,mask_columns] 

f_mean=np.delete(f_mean,abs(pec_edge_element_no),0) 


K_i_bc_real=[]
for kk in range(No_of_dielectric_layer):  
    K_matrix_csr=K_i_real_mean[kk].tocsr()
    mask_rows=np.ones(K_matrix_csr.shape[0],dtype=bool)
    mask_columns=np.ones(K_matrix_csr.shape[1],dtype=bool) 
    mask_rows[abs(pec_edge_element_no)]=False 
    mask_columns[abs(pec_edge_element_no)]=False
    K_matrix_csr=K_matrix_csr[mask_rows]
    K_matrix_csr=K_matrix_csr[:,mask_columns]
    K_i_bc_real.append(K_matrix_csr) 

U_POD_sparse=sparse.lil_matrix(U_POD)
U_POD_sparse_csr=U_POD_sparse.tocsr()
# finding reduced basis matrix
K_bar_sparse_csr=(U_POD_sparse_csr.transpose().dot(K_mean_sparse_csr)).dot(U_POD_sparse_csr)
M_bar_sparse_csr=(U_POD_sparse_csr.transpose().dot(M_mean_sparse_csr)).dot(U_POD_sparse_csr)
B_1_bar_sparse_csr=(U_POD_sparse_csr.transpose().dot(B1_mean_sparse_csr)).dot(U_POD_sparse_csr)
B_2_bar_sparse_csr=(U_POD_sparse_csr.transpose().dot(B2_mean_sparse_csr)).dot(U_POD_sparse_csr)
f_bar=(U_POD.transpose()).dot(f_mean)
K_bar=K_bar_sparse_csr.toarray()
M_bar=M_bar_sparse_csr.toarray()
B_1_bar=B_1_bar_sparse_csr.toarray()
B_2_bar=B_2_bar_sparse_csr.toarray()
K_i_bar_real=[]
for kk in range(No_of_dielectric_layer):
    K_i_bar_sparse=(U_POD_sparse_csr.transpose().dot(K_i_bc_real[kk])).dot(U_POD_sparse_csr)
    K_i_bar_temp=K_i_bar_sparse.toarray()
    K_i_bar_real.append(K_i_bar_temp)
 
# delete unwanted matrix of HDM
del K_mean_sparse_csr
del M_mean_sparse_csr
del B1_mean_sparse_csr
del B2_mean_sparse_csr
del K_i_bc_real 

K_i_bc=[]
for kk in range(No_of_dielectric_layer):
    K_matrix=K_i_bar_real[kk]
    K_i_bc.append(K_matrix)

an_cross_an_cross_Ni_input,e_10_list_input,edge_element_no_list_input,\
        sign_edge_element_list_input,sign_element_no_list_input,\
        sign_element_list_input,Ae_list_input\
    =gte.generate_an_cross_an_cross_Ni(input_face_surface_element_no\
    ,input_face_surface,n,sign_edge_element,input_normal_vector,\
    edge_element_no_input,Ne,node_list,width) 
an_cross_an_cross_Ni_output,e_10_list_output,edge_element_no_list_output,\
        sign_edge_element_list_output,sign_element_no_list_output,\
        sign_element_list_output,Ae_list_output\
    =gte.generate_an_cross_an_cross_Ni(output_face_surface_element_no\
    ,output_face_surface,n,sign_edge_element,output_normal_vector,\
    edge_element_no_output,Ne,node_list,width)  

#poly_order=2

poly_order=2
no_of_hermite_poly=math.factorial(poly_order+No_of_rv)\
       /(math.factorial(poly_order)*math.factorial(No_of_rv))
no_of_hermite_poly=int(no_of_hermite_poly)
Psi_matrix = op.get_hermite_ortho_polynomial_poly_ord_two(no_of_hermite_poly,N_SSFEM,xi_value) 
Psi_j_square=np.zeros((no_of_hermite_poly,no_of_hermite_poly)) 
for ii in range(no_of_hermite_poly):
    for jj in range(no_of_hermite_poly):
        Psi_j_square[ii][jj]=pj.get_Psi_j_square(Psi_matrix[:,ii],Psi_matrix[:,jj]) 
No_of_edges_final=No_of_edges-len(pec_edge_element_no) 
mag_gamma_fem_SSFEM_freq_sweep_ROM=np.zeros((N_SSFEM,len(freq_SSFEM)))
tau_fem_SSFEM_freq_sweep_ROM=np.zeros((N_SSFEM,len(freq_SSFEM)))
for kk_2 in range(len(freq_SSFEM)):  
    omega=2*cmath.pi*freq_SSFEM[kk_2]
    k0=2*math.pi*freq_SSFEM[kk_2]*math.sqrt(mu_0*epsilon_0)   
     
    if k0**2 >=(math.pi/a)**2:
        k_z_10=math.sqrt(k0**2-(math.pi/a)**2) 
    else:
        k_z_10=math.sqrt((math.pi/a)**2-k0**2) 
    f_1_omega=k_z_10*i
    z_value=min(z)
    f_2_omega=k_z_10*cmath.exp(-i*k_z_10*z_value)   
    
    dim_reduced_basis=m_count
    K_stochastic_matrix =sparse.lil_matrix((dim_reduced_basis*no_of_hermite_poly,dim_reduced_basis*no_of_hermite_poly),dtype=complex)
    f_stochastic_matrix=np.zeros(dim_reduced_basis*no_of_hermite_poly,dtype=complex) 

        
    for kk in range(no_of_hermite_poly):
        print(kk)
        for jj in range(no_of_hermite_poly):
            print(jj)
            K_temp=M_bar*Psi_j_square[jj][kk] 
            Psi_ijk_value=np.zeros(No_of_rv)
            for ii in range(No_of_rv):
                Psi_ijk_value[ii]= \
                    pj.get_Psi_i_Psi_j_Psi_k(Psi_matrix[:,ii+1],\
                           Psi_matrix[:,jj],Psi_matrix[:,kk])  
            for ii in range(No_of_rv):
                K_temp=K_temp+ K_i_bc[ii]*Psi_ijk_value[ii]
            K_temp=K_bar*Psi_j_square[jj][kk]+K_temp*(omega**2)+\
                (B_1_bar+B_2_bar)*Psi_j_square[jj][kk]*f_1_omega
            row_value=np.arange(kk*dim_reduced_basis,(kk+1)*dim_reduced_basis)
            col_value=np.arange(jj*dim_reduced_basis,(jj+1)*dim_reduced_basis)
            for row_index in range(len(row_value)): 
                K_stochastic_matrix[row_value[row_index],col_value]=\
                        K_temp[row_index,:]     
    
    index_val=np.arange(0,dim_reduced_basis)
    f_stochastic_matrix[index_val]=f_bar*f_2_omega
    # sparse solve
    K_stochastic_matrix=K_stochastic_matrix.tocsr()
    u_stochastic=spsolve(K_stochastic_matrix,f_stochastic_matrix)
    
    # K_stochastic_matrix_dense=K_stochastic_matrix.toarray(): To convert into array
    # u_stochastic_validation=np.linalg.solve(K_stochastic_matrix_dense,f_stochastic_matrix)
    
    
    # Adding the unknown quantity obtained with the known quantity
    u_stochastic_final=np.zeros(No_of_edges*no_of_hermite_poly,dtype=complex)
    for kk in range(no_of_hermite_poly):
        print(kk)
        count=0
        edge_count=0
        u_stochastic_temp=np.zeros(No_of_edges,dtype=complex)
        u_final=U_POD.dot(u_stochastic[kk*dim_reduced_basis:(kk+1)*dim_reduced_basis]) 
        for ii in range(No_of_edges):
            if edge_count<len(pec_edge_element_no):
                if(ii==abs(pec_edge_element_no[edge_count]) ):
                    u_stochastic_temp[ii]=0  
                    edge_count=edge_count+1 
                else:
                    u_stochastic_temp[ii]=u_final[count]
                    count=count+1  
            else:
                u_stochastic_temp[ii]=u_final[count]
                count=count+1  
        u_stochastic_final[kk*No_of_edges:(kk+1)*No_of_edges]=u_stochastic_temp 
    
     
    gamma_port1_final=np.zeros(no_of_hermite_poly,dtype=complex)
    tau_port2_final=np.zeros(no_of_hermite_poly,dtype=complex) 
    for kk_1 in range(no_of_hermite_poly): 
        E_field_final=u_stochastic_final[kk_1*No_of_edges:(kk_1+1)*No_of_edges]
        gamma_port1_POD=gte.perform_surface_integral(an_cross_an_cross_Ni_input,\
                e_10_list_input,edge_element_no_list_input, sign_edge_element_list_input,\
                sign_element_no_list_input,sign_element_list_input,Ae_list_input,\
                E_field_final,len(input_face_surface_element_no))
        gamma_port1_POD=(gamma_port1_POD*(2*cmath.exp(-i*k_z_10*input_len))/(width*height*E0))
        tau_port2_POD=gte.perform_surface_integral(an_cross_an_cross_Ni_output,\
                e_10_list_output,edge_element_no_list_output, sign_edge_element_list_output,\
                sign_element_no_list_output,sign_element_list_output,Ae_list_output,\
                E_field_final,len(output_face_surface_element_no)) 
        tau_port2_POD=(tau_port2_POD*(2*cmath.exp( i*k_z_10*output_len))/(width*height*E0))  
        gamma_port1_final[kk_1]=gamma_port1_POD 
        tau_port2_final[kk_1]=tau_port2_POD   
    u_SSFEM_ref=np.zeros(N_SSFEM) 
    u_SSFEM_trans=np.zeros(N_SSFEM)
    for jj in range(no_of_hermite_poly):
        u_SSFEM_ref=u_SSFEM_ref+gamma_port1_final[jj]*Psi_matrix[:,jj]
        u_SSFEM_trans=u_SSFEM_trans+tau_port2_final[jj]*Psi_matrix[:,jj]
      
    gamma_fem=u_SSFEM_ref -cmath.exp(-2*i*k_z_10*input_len)
    mag_gamma_fem_SSFEM_freq_sweep_ROM[:,kk_2]=abs(gamma_fem)   
    tau_fem=u_SSFEM_trans
    tau_fem_SSFEM_freq_sweep_ROM[:,kk_2]=abs(tau_fem) 
time_elapsed_SSFEM_ROM=(time.clock()-time_start) 
 
 
    
import time_eval as te
hr,mm,sec=te.time_in_hr_mm_sec(time_elapsed_SSFEM_ROM)
print(hr)
print(mm)
print(sec)

  