/*************************************************************************
	> File Name: SlepcGCGEigen.h
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: Mon Jan  8 19:41:40 2018
 ************************************************************************/


#include <petscksp.h>
#include <petscblaslapack.h>
#include <petsctime.h>
#include <petsc/private/vecimpl.h>
#include "SlepcCG.h"

#define REORTH_TOL 0.75 
#define ORTH_ZERO_TOL 1e-18 

#define A_PRODUCT 0
//#define DEBUG_DETAIL 1


PetscErrorCode GCG_Eigen(Mat A, Mat B, PetscReal *eval, Vec *evec, PetscInt nev, PetscReal abs_tol, PetscInt max_iter);

//用Petsc的矩阵和向量操作构造的函数
PetscErrorCode GetRandomInitValue(Vec *V, PetscInt dim_x);
PetscErrorCode AllocateVecs(Mat A, Vec **evec, Vec **V_1, Vec **V_2, Vec **V_3, PetscInt nev, PetscInt a, PetscInt b, PetscInt c);
PetscErrorCode VecsMatrixVecsForRayleighRitz(Mat A, Vec *V, PetscReal *AA, PetscInt start, PetscInt dim, Vec tmp, DEBUG_PARA *Debug_Para);
PetscErrorCode VecsMatrixVecs(Vec *VL, Mat A, Vec *VR, PetscReal *AA, PetscInt nl, PetscInt nr, Vec tmp);
PetscErrorCode RayleighRitz(Mat A, Mat B, Vec *V, PetscReal *AA, PetscReal *approx_eval, PetscReal *AA_sub, PetscReal *AA_copy, PetscInt start, PetscInt last_dim, PetscInt dim, PetscInt dim_x, Vec tmp, PetscReal *small_tmp, DEBUG_PARA *Debug_Para);
PetscErrorCode GetRitzVectors(PetscReal *SmallEvec, Vec *V, Vec *RitzVec, PetscInt dim, PetscInt n_vec, DEBUG_PARA *Debug_Para, PetscInt if_time);
void ChangeVecPointer(Vec *V_1, Vec *V_2, Vec *tmp, PetscInt size);
PetscErrorCode SumSeveralVecs(Vec *V, PetscReal *x, Vec U, PetscInt n_vec, DEBUG_PARA *Debug_Para);
PetscErrorCode GCG_Orthogonal(Vec *V, Mat B, PetscInt start, PetscInt *end, Vec *V_tmp, Vec *Nonzero_Vec, PetscInt *Ind, DEBUG_PARA *Debug_Para);
PetscReal VecMatrixVec(Vec a, Mat Matrix, Vec b, Vec temp, DEBUG_PARA *Debug_Para);

//小规模的向量或稠密矩阵操作，这些应该是串行的，所以没有改动
void OrthogonalSmall(PetscReal *V, PetscReal **B, PetscInt dim_xpw, PetscInt dim_x, PetscInt *dim_xp, PetscInt *Ind);
void DenseMatVec(PetscReal *DenseMat, PetscReal *x, PetscReal *b, PetscInt dim);
void DenseVecsMatrixVecs(PetscReal *LVecs, PetscReal *DenseMat, PetscReal *RVecs, PetscReal *ProductMat, PetscInt nl, PetscInt nr, PetscInt dim, PetscReal *tmp);
void ScalVecSmall(PetscReal alpha, PetscReal *a, PetscInt n);
PetscReal NormVecSmall(PetscReal *a, PetscInt n);
PetscReal VecDotVecSmall(PetscReal *a, PetscReal *b, PetscInt n);
void SmallAXPBY(PetscReal alpha, PetscReal *a, PetscReal beta, PetscReal *b, PetscInt n);
void SortEigen(PetscReal *evec, PetscReal *eval, PetscInt dim, PetscInt dim_x);


//PetscErrorCode LinearSolverCreate(KSP *ksp, Mat A, Mat T);
PetscErrorCode LinearSolverCreate(KSP *ksp, Mat A, Mat T, DEBUG_PARA *Debug_Para);
//PetscErrorCode LinearSolverCreate(KSP *ksp, Mat A, Mat T, PetscReal cg_tol, PetscInt nsmooth);
PetscErrorCode ComputeAxbResidual(Mat A, Vec x, Vec b, Vec tmp, PetscReal *res);

PetscErrorCode GetLAPACKMatrix(Mat A, Vec *V, PetscReal *AA, PetscReal *AA_sub, PetscInt start, PetscInt last_dim, PetscInt dim, PetscReal *AA_copy, Vec tmp, PetscReal *small_tmp, DEBUG_PARA *Debug_Para);
PetscErrorCode GetWinV(PetscInt start, PetscInt nunlock, PetscInt *unlock, Vec *V, PetscReal *approx_eval, Mat A, Mat B, KSP ksp, PetscReal *RRes, Vec *V_tmp, DEBUG_PARA *Debug_Para);
PetscErrorCode CheckConvergence(Mat A, Mat B, PetscInt *unlock, PetscInt *nunlock, PetscInt nev, Vec *X_tmp, PetscReal *approx_eval, PetscReal abs_tol, Vec *V_tmp, PetscInt iter, PetscReal *RRes, DEBUG_PARA *Debug_Para);
PetscErrorCode GetPinV(PetscReal *AA, Vec *V, PetscInt dim_x, PetscInt last_dim_x, PetscInt *dim_xp, PetscInt dim_xpw, PetscInt nunlock, PetscInt *unlock, Vec *V_tmp, Vec *Orth_tmp, PetscInt *Ind, DEBUG_PARA *Debug_Para);
void GetXinV(Vec *V, Vec *X_tmp, Vec *tmp, PetscInt dim_x);
void Updatedim_x(PetscInt start, PetscInt end, PetscInt *dim_x, PetscReal *approx_eval);
void PrintSmallEigen(PetscInt iter, PetscInt nev, PetscReal *approx_eval, PetscReal *Rayleigh, PetscReal *AA, PetscInt dim, PetscReal *RRes);

PetscInt petscmax(PetscInt a, PetscInt b);
void CreateDEBUG_PARA(DEBUG_PARA **Debug_Para);
void FreeDEBUG_PARA(DEBUG_PARA *Debug_Para);
