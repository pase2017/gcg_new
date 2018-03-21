/*************************************************************************
	> File Name: SlepcCG.h
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: Mon Jan  8 19:39:34 2018
 ************************************************************************/

#include <petscksp.h>
#include <petsctime.h>
#define EPS 2.220446e-16
#define DEBUG_DETAIL 1
#define DEBUG_TIME 1
#define USE_KSP 1
//#define EPS 1e-10

typedef struct DEBUG_PARA_ {
    PetscReal One_GetW_Time;
    PetscReal One_GetX_Time;
    PetscReal One_GetP_Time;
    PetscReal One_Orth_Time;
    PetscReal One_Rayl_Time;
    PetscReal One_Conv_Time;
    PetscReal GetW_Time;
    PetscReal GetX_Time;
    PetscReal GetP_Time;
    PetscReal Orth_Time;
    PetscReal Rayl_Time;
    PetscReal Conv_Time;
    PetscReal CG_Time;
    PetscInt  CG_Iter;
    PetscReal SPMV_Time;
    PetscInt  SPMV_Iter;
    PetscReal CGSPMV_Time;
    PetscInt  CGSPMV_Iter;
    PetscReal VDot_Time;
    PetscInt  VDot_Iter;
    PetscReal CGVDot_Time;
    PetscInt  CGVDot_Iter;
    PetscReal Norm_Time;
    PetscInt  Norm_Iter;
    PetscReal CGNorm_Time;
    PetscInt  CGNorm_Iter;
    PetscReal AXPY_Time;
    PetscInt  AXPY_Iter;
    PetscReal Ritz_Time;
    PetscReal Small_Time;
    PetscInt  cg_max_it;
    PetscReal cg_rate;
}DEBUG_PARA;

PetscErrorCode SLEPCCG(Mat Matrix, Vec b, Vec x, PetscReal accur, PetscInt Max_It, Vec *V_tmp);
PetscErrorCode SLEPCCG_2(Mat Matrix, Vec b, Vec x, PetscReal rate, PetscInt Max_It, Vec *V_tmp, DEBUG_PARA *Debug_Para);
