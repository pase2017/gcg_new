/*************************************************************************
	> File Name: ReadWritePrint.h
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: Mon Jan  8 19:40:30 2018
 ************************************************************************/

#include <petscksp.h>
#include <petscblaslapack.h>
#include <petsctime.h>
#include <petsc/private/vecimpl.h>

PetscErrorCode ReadScaleMatrixBinary(Mat *A);
PetscErrorCode ReadPetscMatrixBinary(Mat *A, const char *filename);
PetscErrorCode ReadGenaralMatrixBinary(Mat *A, Mat *B, PetscInt ifBI);
PetscErrorCode GenerateFDMatrix(Mat *A, Mat *B);
PetscErrorCode ReadMatrix(const char *filename, Mat *A);
PetscErrorCode ReadCSRMatrix(Mat *A, Mat *B);
PetscErrorCode ReadVec(Vec *V, PetscInt nev);
PetscErrorCode ReadVecformfile(char *filename, Vec *V, PetscInt nev);
PetscErrorCode PrintVec(Vec *V, PetscInt n_vec, PetscInt start, PetscInt end);
PetscErrorCode WriteVec(Vec *V, PetscInt nev);
PetscErrorCode WriteVectofile(char *filename, Vec *V, PetscInt n_vec);
PetscErrorCode PrintVecparallel(Mat A, Vec *V, PetscInt n_vec, PetscInt flag);
