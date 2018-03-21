/*************************************************************************
	> File Name: SlepcCG.c
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: Mon Jan  8 19:30:40 2018
 ************************************************************************/
#include "SlepcCG.h"

PetscErrorCode SLEPCCG(Mat Matrix, Vec b, Vec x, PetscReal accur, PetscInt Max_It, Vec *V_tmp)
{
    Vec            r, p, tmp;
    PetscReal      alpha, beta, error, init_error, tmp1, tmp2;
    PetscInt       niter = 0;
    PetscErrorCode ierr;
	r   = V_tmp[0];
	p   = V_tmp[1];
	tmp = V_tmp[2];
  
    ierr = MatMult(Matrix,x,p);CHKERRQ(ierr); //tmp1 = A*x0 
 
    ierr = VecAXPBYPCZ(r, 1.0, -1.0, 0.0, b, p);  //r = b - tmp1
	//printf("in CG! the initial residual norm: %lf\n", NormVec(r));
    ierr = VecCopy(r, p); CHKERRQ(ierr);//p=r
    ierr = VecNorm(r, 1, &init_error); CHKERRQ(ierr);
    do{
        ierr = MatMult(Matrix,p,tmp);CHKERRQ(ierr);//tmp = A*p
        //PetscPrintf(PETSC_COMM_WORLD, "matrix*p=tmp:\n");
        //ierr = PrintVec(&p, 1, 10, 20);
        ierr = VecDot(r, p, &tmp1);CHKERRQ(ierr);
        ierr = VecDot(p, tmp, &tmp2);CHKERRQ(ierr);
        alpha = tmp1/tmp2;
        //PetscPrintf(PETSC_COMM_WORLD, "alpha: %18.15lf\n", alpha);
        ierr = VecAXPY(x, alpha, p);CHKERRQ(ierr);
        ierr = VecAXPY(r, -alpha, tmp);CHKERRQ(ierr);
        ierr = VecNorm(r, 1, &error); CHKERRQ(ierr);
    
        if(error/init_error<accur)
            break;
    
        ierr = VecDot(r, tmp, &tmp1);CHKERRQ(ierr);
        ierr = VecDot(p, tmp, &tmp2);CHKERRQ(ierr);
        beta = -tmp1/tmp2;
        //PetscPrintf(PETSC_COMM_WORLD, "beta: %18.15lf\n", beta);
        //beta = -VecDotVec(r,tmp) / VecDotVec(p,tmp); //beta = -(r,tmp)/(p,tmp)
   
        ierr = VecAYPX(p, beta, r); CHKERRQ(ierr);
		//if(niter%100 == 0)
		//	printf("in CG, niter = %d, residual:%15.5e\n", niter, error);
        niter++; 
    }while((error >= EPS)&&(niter<Max_It));

    //printf("CG: iterations: %3d, residual  :%18.15e, time: %10.5lfs, dof_all:%d\n", niter, error, t2-t1, Matrix->N_Rows);
    //printf("end of CG solving!\n");
    PetscFunctionReturn(0);
}

PetscErrorCode SLEPCCG_2(Mat Matrix, Vec b, Vec x, PetscReal rate, PetscInt Max_It, Vec *V_tmp, DEBUG_PARA *Debug_Para)
{
    PetscErrorCode ierr;
#if DEBUG_TIME
    PetscLogDouble t1, t2;
    ierr = PetscTime(&t1);CHKERRQ(ierr);
#endif
    Vec            r, p, tmp;
    PetscReal      alpha, beta, rho, rho_1, tol = EPS, bnrm2, error, init_error, tmp2;
    PetscInt       niter = 0, print_cg = 0, CG_Iter;
	r   = V_tmp[0];
	p   = V_tmp[1];
	tmp = V_tmp[2];
    ierr = PetscOptionsGetInt(NULL, NULL, "-print_cg", &print_cg, NULL); CHKERRQ(ierr);
    //printf("print_cg=%d\n", print_cg);
  
#if DEBUG_DETAIL
    PetscLogDouble tmv1, tmv2;
    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
    ierr = VecNorm(b, 1, &bnrm2); CHKERRQ(ierr);
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
    Debug_Para->Norm_Time += (double)(tmv2-tmv1);
    Debug_Para->Norm_Iter += 1;
    Debug_Para->CGNorm_Time += (double)(tmv2-tmv1);
    Debug_Para->CGNorm_Iter += 1;
#endif
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
    ierr = MatMult(Matrix,x,p);CHKERRQ(ierr); //tmp1 = A*x0 
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
    Debug_Para->SPMV_Time += (double)(tmv2-tmv1);
    Debug_Para->SPMV_Iter += 1;
    Debug_Para->CGSPMV_Time += (double)(tmv2-tmv1);
    Debug_Para->CGSPMV_Iter += 1;
#endif
 
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
    ierr = VecAXPBYPCZ(r, 1.0, -1.0, 0.0, b, p);  //r = b - tmp1
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
    Debug_Para->AXPY_Time += (double)(tmv2-tmv1);
    Debug_Para->AXPY_Iter += 2;
#endif
	//printf("in CG! the initial residual norm: %lf\n", NormVec(r));
    //ierr = VecCopy(r, p); CHKERRQ(ierr);//p=r
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
    ierr = VecNorm(r, 1, &init_error); CHKERRQ(ierr);
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
    Debug_Para->Norm_Time += (double)(tmv2-tmv1);
    Debug_Para->Norm_Iter += 1;
    Debug_Para->CGNorm_Time += (double)(tmv2-tmv1);
    Debug_Para->CGNorm_Iter += 1;
#endif
    error = init_error/bnrm2;
    CG_Iter = Max_It;
    if(error < tol)
        PetscFunctionReturn(0);
    for( niter=0; niter<Max_It; niter++ )
    {
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
        ierr = VecDot(r, r, &rho);CHKERRQ(ierr);
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
        Debug_Para->VDot_Time += (double)(tmv2-tmv1);
        Debug_Para->VDot_Iter += 1;
        Debug_Para->CGVDot_Time += (double)(tmv2-tmv1);
        Debug_Para->CGVDot_Iter += 1;
#endif
        if(niter > 0)
        {
            beta = rho/rho_1;
            ierr = VecAYPX(p, beta, r); CHKERRQ(ierr);
        }
        else
        {
            ierr = VecCopy(r, p); CHKERRQ(ierr);//p=r
        }
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
        ierr = MatMult(Matrix,p,tmp);CHKERRQ(ierr);//tmp = A*p
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
        Debug_Para->SPMV_Time += (double)(tmv2-tmv1);
        Debug_Para->SPMV_Iter += 1;
        Debug_Para->CGSPMV_Time += (double)(tmv2-tmv1);
        Debug_Para->CGSPMV_Iter += 1;
#endif
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
        ierr = VecDot(p, tmp, &tmp2);CHKERRQ(ierr);
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
        Debug_Para->VDot_Time += (double)(tmv2-tmv1);
        Debug_Para->VDot_Iter += 1;
        Debug_Para->CGVDot_Time += (double)(tmv2-tmv1);
        Debug_Para->CGVDot_Iter += 1;
#endif
        alpha = rho/tmp2;
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
        ierr = VecAXPY(x, alpha, p);CHKERRQ(ierr);
        ierr = VecAXPY(r, -alpha, tmp);CHKERRQ(ierr);
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
        Debug_Para->AXPY_Time += (double)(tmv2-tmv1);
        Debug_Para->AXPY_Iter += 2;
#endif
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
        ierr = VecNorm(r, 1, &error); CHKERRQ(ierr);
#if DEBUG_DETAIL
	ierr = PetscTime(&tmv2);CHKERRQ(ierr);
	Debug_Para->Norm_Time += (double)(tmv2-tmv1);
	Debug_Para->Norm_Iter += 1;
	Debug_Para->CGNorm_Time += (double)(tmv2-tmv1);
	Debug_Para->CGNorm_Iter += 1;
#endif
        if(print_cg == 1)
            PetscPrintf(PETSC_COMM_WORLD, "in CG, iter: %d, error: %18.15lf\n", niter, error);
    
        if((error/init_error<rate)||(error/bnrm2<tol))
        {
            CG_Iter = niter + 1;
            break;
        }
        rho_1 = rho;
        //niter++; 
    }

    //printf("CG: iterations: %3d, residual  :%18.15e, time: %10.5lfs, dof_all:%d\n", niter, error, t2-t1, Matrix->N_Rows);
    //printf("end of CG solving!\n");
#if DEBUG_DETAIL
    ierr = PetscTime(&t2);CHKERRQ(ierr);
    Debug_Para->CG_Time += (double)(t2-t1);
    Debug_Para->CG_Iter += CG_Iter;
#endif
    PetscFunctionReturn(0);
}
