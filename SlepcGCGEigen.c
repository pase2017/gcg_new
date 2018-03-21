/*************************************************************************
	> File Name: SlepcGCGEigen.c
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: Mon Jan  8 19:37:31 2018
 ************************************************************************/

#include "SlepcGCGEigen.h"
#include "ReadWritePrint.h"

PetscErrorCode GCG_Eigen(Mat A, Mat B, PetscReal *eval, Vec *evec, PetscInt nev, PetscReal abs_tol, PetscInt max_iter)
{
    //--------------------定义变量--------------------------------------------
    PetscInt       i, max_dim_x = nev*5/4;//最大是1.25×nev
    PetscLogDouble ttotal1, ttotal2, titer1, titer2;
    //unlock用来记录没有收敛的特征值和特征向量在V中的编号,nunlock为未收敛的特征对个数
    //dim_xpw表示V的长度,dim_xp表示[X,P]的向量个数,dim_x表示X中的向量个数
    PetscInt       *unlock, nunlock, dim_xpw, last_dim_xpw, dim_xp, dim_x = nev, last_dim_x = dim_x, iter = 0, *Ind;
    //AA_copy用来存储矩阵AA的备份，为了下次计算做准备
    //AA_sub用来存储小规模矩阵AA中与[X,P]对应的对角部分
    PetscReal      *AA, *approx_eval, *AA_copy, *AA_sub, *small_tmp, *RRes;// *time, *last_time;
    Vec            *V, *V_tmp, *X_tmp, *Orth_tmp;
    PetscErrorCode ierr;
    DEBUG_PARA     *Debug_Para;
    CreateDEBUG_PARA(&Debug_Para);

    ierr = PetscTime(&ttotal1);CHKERRQ(ierr);
    //--------------------分配空间--------------------------------------------
    //给V，Vtmp,x2分配空间,V用于存储[X,d,W],Vtmp是临时存储空间,x2用于存储近似Ritz向量
    ierr = AllocateVecs(A, &evec, &V, &V_tmp, &X_tmp, nev, 3*max_dim_x, petscmax(3*max_dim_x, 4), max_dim_x);CHKERRQ(ierr);
    //给小规模特征值计算的变量分配空间,small_tmp是临时存储空间,计算AA_sub时用
    approx_eval = (PetscReal*)calloc(3*max_dim_x, sizeof(PetscReal));
    AA          = (PetscReal*)calloc(9*max_dim_x*max_dim_x, sizeof(PetscReal));
    AA_copy     = (PetscReal*)calloc(9*max_dim_x*max_dim_x, sizeof(PetscReal));
    AA_sub      = (PetscReal*)calloc(4*max_dim_x*max_dim_x, sizeof(PetscReal));
    small_tmp   = (PetscReal*)calloc(3*max_dim_x, sizeof(PetscReal));
    unlock      = calloc(nev, sizeof(PetscInt));
    Ind         = calloc(3*max_dim_x, sizeof(PetscInt));
    RRes        = (PetscReal*)calloc(nev, sizeof(PetscReal));
    //time用于记录各部分时间,其中time[0]:cg计算w的时间,time[1]:正交化的时间,
    //time[2]:Rayleigh-Ritz的时间,time[3]:计算Ritz向量X的时间,time[4]:计算P的时间
    //time        = (PetscReal*)calloc(5, sizeof(PetscReal));
    //last_time   = (PetscReal*)calloc(5, sizeof(PetscReal));
    Orth_tmp    = (Vec*)malloc(max_dim_x*sizeof(Vec));

    ierr = PetscOptionsGetReal(NULL,NULL,"-cg_rate",&(Debug_Para->cg_rate),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-cg_max_it",&(Debug_Para->cg_max_it),NULL);CHKERRQ(ierr);
    KSP ksp;
#if USE_KSP

    Mat T;
    PetscInt scale_type = 2;
    ierr = PetscOptionsGetInt(NULL,NULL,"-scale_type",&scale_type,NULL);CHKERRQ(ierr);
    if( scale_type == 1 )
    {
        PetscPrintf(PETSC_COMM_WORLD, "scale_type = 1, use T = A. \n");
        T = A;
    }
    else if( scale_type == 2 )
    {
        PetscPrintf(PETSC_COMM_WORLD, "scale_type = 2, use T read from file. \n");
        ierr = ReadScaleMatrixBinary(&T);
    }

    ierr = LinearSolverCreate(&ksp, A, T, Debug_Para);
#endif
   

    //------------------开始CGC计算特征值--------------------------------
    //对初值做一次B正交化
#if A_PRODUCT
    PetscPrintf(PETSC_COMM_WORLD, "A product!\n");
    ierr = GCG_Orthogonal(V, A, 0, &dim_x, V_tmp, Orth_tmp, Ind, Debug_Para);CHKERRQ(ierr);
#else
    ierr = GCG_Orthogonal(V, B, 0, &dim_x, V_tmp, Orth_tmp, Ind, Debug_Para);CHKERRQ(ierr);
#endif
    //计算得到小规模特征值计算的矩阵AA,BB并保存备份AA_copy
    ierr = RayleighRitz(A, B, V, AA, approx_eval, NULL, AA_copy, 0, 0, dim_x, dim_x, V_tmp[0], NULL, Debug_Para);CHKERRQ(ierr);
#if DEBUG_DETAIL
    PetscLogDouble tmv1, tmv2;
    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
    ierr = MPI_Bcast(AA, dim_x*dim_x, MPI_DOUBLE, 0, PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = MPI_Bcast(approx_eval, dim_x, MPI_DOUBLE, 0, PETSC_COMM_WORLD);CHKERRQ(ierr);
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
    Debug_Para->Small_Time += (double)(tmv2-tmv1);
#endif
    //计算初始近似特征向量并正交化
    ierr = GetRitzVectors(AA, V, X_tmp, dim_x, dim_x, Debug_Para, 1);CHKERRQ(ierr);
    ChangeVecPointer(V, X_tmp, Orth_tmp, dim_x);
    ierr = CheckConvergence(A, B, unlock, &nunlock, nev, V, approx_eval, abs_tol, V_tmp, -1, RRes, Debug_Para);CHKERRQ(ierr);
    //用CG迭代获取W向量
    ierr = GetWinV(dim_x, nunlock, unlock, V, approx_eval, A, B, ksp, RRes, V_tmp, Debug_Para);CHKERRQ(ierr);
    //对V进行正交化,并计算evec=V^T*A*V,B1=V^T*B*V
    dim_xpw = 2*dim_x;
#if A_PRODUCT
    ierr = GCG_Orthogonal(V, A, dim_x, &dim_xpw, V_tmp, Orth_tmp, Ind, Debug_Para);CHKERRQ(ierr);
#else
    ierr = GCG_Orthogonal(V, B, dim_x, &dim_xpw, V_tmp, Orth_tmp, Ind, Debug_Para);CHKERRQ(ierr);
#endif
    ierr = RayleighRitz(A, B, V, AA, approx_eval, NULL, AA_copy, 0, 0, dim_xpw, dim_x, V_tmp[0], NULL, Debug_Para);CHKERRQ(ierr);
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
    ierr = MPI_Bcast(AA, dim_x*dim_xpw, MPI_DOUBLE, 0, PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = MPI_Bcast(approx_eval, dim_x, MPI_DOUBLE, 0, PETSC_COMM_WORLD);CHKERRQ(ierr);
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
    Debug_Para->Small_Time += (double)(tmv2-tmv1);
#endif
    //计算Ritz向量,由于得到的特征向量是关于B正交的，不需要对x2进行正交化
    ierr = GetRitzVectors(AA, V, X_tmp, dim_xpw, dim_x, Debug_Para, 1);CHKERRQ(ierr);
    ierr = CheckConvergence(A, B, unlock, &nunlock, nev, X_tmp, approx_eval, abs_tol, V_tmp, 0, RRes, Debug_Para);CHKERRQ(ierr);

    //--------------------开始循环--------------------------------------------
    while((nunlock > 0)&&(iter < max_iter))
    {
        //memcpy(last_time, time, 5*sizeof(PetscReal));
        ierr = PetscTime(&titer1);CHKERRQ(ierr);
        //更新dim_xp,dim_xp表示[X,P]的向量个数
        dim_xp = dim_x+nunlock;
        //计算P
        ierr = GetPinV(AA, V, dim_x, last_dim_x, &dim_xp, dim_xpw, nunlock, unlock, V_tmp, Orth_tmp, Ind, Debug_Para);CHKERRQ(ierr);
        GetXinV(V, X_tmp, Orth_tmp, dim_x);
        //更新dim_xpw为V=[X,P,W]的向量个数
        last_dim_xpw = dim_xpw;
        dim_xpw = dim_xp+nunlock;
        //对unlock的X进行CG迭代得到W,V的前nev列为X,dim_xp列之后是W
        ierr = GetWinV(dim_xp, nunlock, unlock, V, approx_eval, A, B, ksp, RRes, V_tmp, Debug_Para);CHKERRQ(ierr);
        //对W与前dim_xp个向量进行正交化,Ind记录W中的非零向量的列号
#if A_PRODUCT
        ierr = GCG_Orthogonal(V, A, dim_xp, &dim_xpw, V_tmp, Orth_tmp, Ind, Debug_Para);CHKERRQ(ierr);
#else
        ierr = GCG_Orthogonal(V, B, dim_xp, &dim_xpw, V_tmp, Orth_tmp, Ind, Debug_Para);CHKERRQ(ierr);
        //ierr = GCG_Orthogonal(V, B, 0, &dim_xpw, V_tmp, Orth_tmp, Ind, Debug_Para);CHKERRQ(ierr);
#endif
        //计算小规模矩阵特征值
        ierr = RayleighRitz(A, B, V, AA, approx_eval, AA_sub, AA_copy, dim_xp, last_dim_xpw, dim_xpw, dim_x, V_tmp[0], small_tmp, Debug_Para);CHKERRQ(ierr);
        //ierr = RayleighRitz(A, B, V, AA, approx_eval, AA_sub, AA_copy, dim_x, last_dim_xpw, dim_xpw, dim_x, V_tmp[0], small_tmp, Debug_Para);CHKERRQ(ierr);
        //检查特征值重数
        last_dim_x = dim_x;
        Updatedim_x(nev, max_dim_x, &dim_x, approx_eval);
#if DEBUG_DETAIL
	ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
	ierr = MPI_Bcast(AA, dim_x*dim_xpw, MPI_DOUBLE, 0, PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Bcast(approx_eval, dim_x, MPI_DOUBLE, 0, PETSC_COMM_WORLD);CHKERRQ(ierr);
#if DEBUG_DETAIL
	ierr = PetscTime(&tmv2);CHKERRQ(ierr);
	Debug_Para->Small_Time += (double)(tmv2-tmv1);
#endif
        //计算Ritz向量
        ierr = GetRitzVectors(AA, V, X_tmp, dim_xpw, dim_x, Debug_Para, 1);CHKERRQ(ierr);
        ierr = CheckConvergence(A, B, unlock, &nunlock, nev, X_tmp, approx_eval, abs_tol, V_tmp, iter+1, RRes, Debug_Para);CHKERRQ(ierr);
        ierr = PetscTime(&titer2);CHKERRQ(ierr);

#if DEBUG_TIME
        PetscPrintf(PETSC_COMM_WORLD,"iter_total(%d) =  %lf; get_w(%d) = %lf; orth(%d) = %lf; rayleigh_ritz(%d) = %lf; get_x(%d) = %lf; get_p(%d) = %lf;\n", 
                iter+1, (double)(titer2-titer1), iter+1, Debug_Para->One_GetW_Time, iter+1, Debug_Para->One_Orth_Time, iter+1, Debug_Para->One_Rayl_Time, iter+1, Debug_Para->One_GetX_Time, iter+1, Debug_Para->One_GetP_Time);
#endif
#if DEBUG_DETAIL
        PetscPrintf(PETSC_COMM_WORLD, "CG time(%d) = %lf; CG_Iter(%d) = %d; SPMV_time(%d) = %lf; SPMV_Iter(%d) = %d;\nVDot_Time(%d) = %lf; VDot_Iter(%d) = %d; AXPY_Time(%d) = %lf; AXPY_Iter(%d) = %d;\n Norm_Time(%d) = %lf; Norm_Iter(%d) = %d; Smal_time(%d) = %lf;\n CGSPMV_Time(%d) = %lf; CGSPMV_Iter(%d) = %d; CGVDot_time(%d) = %lf; CGVDot_Iter(%d) = %d;\n CGNorm_time(%d) = %lf, CGNorm_Iter(%d) = %d;\n",
                iter+1, Debug_Para->CG_Time, iter+1, Debug_Para->CG_Iter,
                iter+1, Debug_Para->SPMV_Time, iter+1, Debug_Para->SPMV_Iter,
                iter+1, Debug_Para->VDot_Time, iter+1, Debug_Para->VDot_Iter,
                iter+1, Debug_Para->AXPY_Time, iter+1, Debug_Para->AXPY_Iter, 
                iter+1, Debug_Para->Norm_Time, iter+1, Debug_Para->Norm_Iter, 
		iter+1, Debug_Para->Small_Time,
                iter+1, Debug_Para->CGSPMV_Time, iter+1, Debug_Para->CGSPMV_Iter,
                iter+1, Debug_Para->CGVDot_Time, iter+1, Debug_Para->CGVDot_Iter,
                iter+1, Debug_Para->CGNorm_Time, iter+1, Debug_Para->CGNorm_Iter);
#endif
        iter += 1;
    }

    //PrintSmallEigen(iter, nev, approx_eval, NULL, NULL, 0, RRes);
    ierr = PetscTime(&ttotal2);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"'total time: %lf, get w: %lf, orth: %lf, rayleigh-ritz: %lf, get x: %lf, get p: %lf'\n", 
            (double)(ttotal2-ttotal1), Debug_Para->GetW_Time, Debug_Para->Orth_Time, Debug_Para->Rayl_Time, Debug_Para->GetX_Time, Debug_Para->GetP_Time);
    memcpy(eval, approx_eval, nev*sizeof(PetscReal));
    for( i=0; i<nev; i++ )
    {
        evec[i] = X_tmp[i];
    }
    //------------GCG迭代结束------------------------------------

    //释放空间
    free(approx_eval);  approx_eval = NULL;
    free(unlock);       unlock      = NULL;
    free(AA_sub);       AA_sub      = NULL;
    free(AA_copy);      AA_copy     = NULL;
    free(AA);           AA          = NULL;
    free(small_tmp);    small_tmp   = NULL;
    free(RRes);         RRes        = NULL;
    free(Ind);          Ind         = NULL;
    //free(time);         time        = NULL;
    //free(last_time);    last_time   = NULL;
    FreeDEBUG_PARA(Debug_Para);
    KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    VecDestroyVecs(3*max_dim_x, &V);
    VecDestroyVecs(petscmax(3*max_dim_x, 4), &V_tmp);
    for( i=nev; i<max_dim_x; i++ )
    {
        VecDestroy(&(X_tmp[i]));
    }
    ierr = PetscFree(X_tmp);CHKERRQ(ierr);
    ierr = PetscFree(Orth_tmp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


PetscErrorCode AllocateVecs(Mat A, Vec **evec, Vec **V_1, Vec **V_2, Vec **V_3, PetscInt nev, PetscInt a, PetscInt b, PetscInt c)
{
    Vec            vec_tmp;
    PetscErrorCode ierr;
    ierr = MatCreateVecs(A,NULL,&vec_tmp);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(vec_tmp,a,V_1);CHKERRQ(ierr);
    PetscInt       i;
    for( i=0; i<nev; i++ )
    {
        ierr = VecCopy((*evec)[i], (*V_1)[i]); CHKERRQ(ierr);
        VecDestroy(&((*evec)[i]));
    }
    ierr = VecDuplicateVecs(vec_tmp,b,V_2);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_tmp); CHKERRQ(ierr);
    ierr = VecDuplicateVecs((*V_2)[0],c,V_3);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

void PrintSmallEigen(PetscInt iter, PetscInt nev, PetscReal *approx_eval, PetscReal *Rayleigh, PetscReal *AA, PetscInt dim, PetscReal *RRes)
{
    PetscInt i, j;
    //PetscPrintf(PETSC_COMM_WORLD, "'in while, the iter: %d LAPACKsyev:'\n", iter);
    //PetscMPIInt rank;
    //PetscErrorCode ierr;
    //ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);//CHKERRQ(ierr);
    if(Rayleigh == NULL)
    {
        for( i=0; i<nev; i++ )
        	PetscPrintf(PETSC_COMM_WORLD, "approx_eval(%d,%d) = %18.15lf; abs_res(%d,%d) =  %18.15e;\n", iter, i+1, approx_eval[i], iter, i+1, RRes[i]);
            //PetscPrintf(PETSC_COMM_WORLD, "'iter: %d, rank: %d, approx_eval[%d] = %18.15lf, abosolute residual: %18.15e'\n", iter, rank, i, approx_eval[i], RRes[i]);
    }
    else
    {
        for( i=0; i<nev; i++ )
        	PetscPrintf(PETSC_COMM_WORLD, "aeval(%d,%d)=%18.15lf;res(%d,%d)= %18.15e;Rval(%d,%d)=%18.15e;Rres(%d,%d)=%18.15e;\n", iter, i+1, approx_eval[i], iter, i+1, RRes[i], iter, i+1, Rayleigh[i]*(1e8), RRes[i]*(1e8));
            //PetscPrintf(PETSC_COMM_WORLD,"'iter: %d, rank: %d, approx_eval[%d] = %18.15lf, abosolute residual: %18.15e, rayleigh: %18.15lf'\n", 
            //        iter, rank, i, approx_eval[i], RRes[i], Rayleigh[i]);
    }
        //PetscPrintf(PETSC_COMM_WORLD, "'approx_eval[%d] = %18.15lf, abosolute residual: %e'\n", i, approx_eval[i], RRes[i]);
    if(AA != NULL)
    {
        for( i=0; i<nev; i++ )
            for( j=0; j<dim; j++ )
                PetscPrintf(PETSC_COMM_WORLD, "small evec[%d][%d] = %18.15lf\n", i, j, AA[i*dim+j]);
    }
}


//计算lapack计算特征值时的小规模矩阵AA,start表示需要计算的起始列号,dim表示小规模矩阵的维数
PetscErrorCode GetLAPACKMatrix(Mat A, Vec *V, PetscReal *AA, PetscReal *AA_sub, PetscInt start, PetscInt last_dim, PetscInt dim, PetscReal *AA_copy, Vec tmp, PetscReal *small_tmp, DEBUG_PARA *Debug_Para)
{
    PetscInt       i;
    PetscErrorCode ierr;
#if DEBUG_DETAIL
    PetscLogDouble tmv1, tmv2;
    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
    //计算AA_sub
    if(start != 0)
    {
        DenseVecsMatrixVecs(NULL, AA_copy, AA, AA_sub, 0, start, last_dim, small_tmp);
    }
    memset(AA, 0.0, dim*dim*sizeof(PetscReal));
    for( i=0; i<start; i++ )
        memcpy(AA+i*dim, AA_sub+i*start, start*sizeof(PetscReal));
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
    Debug_Para->Small_Time += (double)(tmv2-tmv1);
#endif
    ierr = VecsMatrixVecsForRayleighRitz(A, V, AA, start, dim, tmp, Debug_Para);CHKERRQ(ierr);
    //ierr = VecsMatrixVecsForRayleighRitz(A, V, AA, 0, dim, tmp, Debug_Para);CHKERRQ(ierr);
    //ierr = VecsMatrixVecsForRayleighRitz(A, V, AA_copy, 0, dim, tmp, Debug_Para);CHKERRQ(ierr);
    memcpy(AA_copy, AA, dim*dim*sizeof(PetscReal));
    PetscFunctionReturn(0);
}

//用CG迭代得到向量W
PetscErrorCode GetWinV(PetscInt start, PetscInt nunlock, PetscInt *unlock, Vec *V, PetscReal *approx_eval, Mat A, Mat B, KSP ksp, PetscReal *RRes, Vec *V_tmp, DEBUG_PARA *Debug_Para)
{
    PetscInt       i, j, nsmooth = Debug_Para->cg_max_it;
    PetscErrorCode ierr;
#if DEBUG_TIME
    PetscLogDouble t1, t2, tmv1, tmv2;
    ierr = PetscTime(&t1);CHKERRQ(ierr);
#endif
    //ierr = KSPSetOperators(ksp,A,T);CHKERRQ(ierr);
    /*
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, 1);
    KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
    */

    PetscReal      cg_rate = Debug_Para->cg_rate;

    char filename[PETSC_MAX_PATH_LEN] = "fileinput";


    for( i=0; i<nunlock; i++ )
    {
        j = unlock[i];
        //初值V[start+i]=V[i]
        //Vtmp[0]=\lambda*B*V[i]作为右端项
        //计算V[start+i]=A^(-1)BV[i]
        //调用CG迭代来计算线性方程组
        ierr = VecCopy(V[j], V[start+i]); CHKERRQ(ierr);
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
        ierr = MatMult(B, V[j], V_tmp[0]); CHKERRQ(ierr);
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
        Debug_Para->SPMV_Time += (double)(tmv2-tmv1);
        Debug_Para->SPMV_Iter += 1;
#endif
        ierr = VecScale(V_tmp[0], approx_eval[j]); CHKERRQ(ierr);
        //ierr = SLEPCCG(A, V_tmp[0], V[start+i], cg_tol, nsmooth, V_tmp+1);CHKERRQ(ierr); 
#if USE_KSP
    	ierr = KSPSolve(ksp,V_tmp[0],V[start+i]);CHKERRQ(ierr);
	//if (ksp->reason == -8)
	/*
	if(i == 4)
	{
	    ierr = PetscOptionsGetString(NULL, NULL, "-rhs_vec", filename, sizeof(filename), NULL); CHKERRQ(ierr);
	    PetscPrintf(PETSC_COMM_WORLD, "output rhs vec: filename: %s\n", filename);
	    ierr = WriteVectofile(filename, V_tmp, 1);
	    ierr = PetscOptionsGetString(NULL, NULL, "-init_vec", filename, sizeof(filename), NULL); CHKERRQ(ierr);
	    PetscPrintf(PETSC_COMM_WORLD, "output init vec: filename: %s\n", filename);
	    ierr = WriteVectofile(filename, V+j, 1);
	    break;
	}
	*/
#else
        ierr = SLEPCCG_2(A, V_tmp[0], V[start+i], cg_rate, nsmooth, V_tmp+1, Debug_Para);CHKERRQ(ierr);
#endif
    }
#if DEBUG_TIME
    ierr = PetscTime(&t2);CHKERRQ(ierr);
    Debug_Para->GetW_Time += (double)(t2-t1);
    Debug_Para->One_GetW_Time = (double)(t2-t1);
#endif
    //time[0] += (double)(t2-t1);
    PetscFunctionReturn(0);
}

PetscErrorCode ComputeAxbResidual(Mat A, Vec x, Vec b, Vec tmp, PetscReal *res)
{
    PetscErrorCode ierr;
    ierr = MatMult(A, x, tmp); CHKERRQ(ierr);
    ierr = VecAXPY(tmp, -1.0, b); CHKERRQ(ierr);
    ierr = VecNorm(tmp, 1, res); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

//计算残差，并获取未收敛的特征对编号及个数
PetscErrorCode CheckConvergence(Mat A, Mat B, PetscInt *unlock, PetscInt *nunlock, PetscInt nev, Vec *X_tmp, PetscReal *approx_eval, PetscReal abs_tol, Vec *V_tmp, PetscInt iter, PetscReal *RRes, DEBUG_PARA *Debug_Para)
{
    PetscInt       i, nunlocktmp = 0, print_eval = 1;
    PetscReal      res_norm, evec_norm, res, max_res = 0.0, min_res = 0.0, sum_res = 0.0;
    PetscErrorCode ierr;
#if DEBUG_TIME
    PetscLogDouble t1, t2, tmv1, tmv2, tvd1, tvd2;
    ierr = PetscTime(&t1);CHKERRQ(ierr);
#endif
    //PetscReal      *Rayleigh = (PetscReal*)calloc(nev,sizeof(PetscReal));
    for( i=0; i<nev; i++ )
    {
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
        ierr = MatMult(A, X_tmp[i], V_tmp[0]); CHKERRQ(ierr);
        ierr = MatMult(B, X_tmp[i], V_tmp[1]); CHKERRQ(ierr);
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
        Debug_Para->SPMV_Time += (double)(tmv2-tmv1);
        Debug_Para->SPMV_Iter += 2;
#endif

#if DEBUG_DETAIL
        ierr = PetscTime(&tvd1);CHKERRQ(ierr);
#endif
        ierr = VecDot(X_tmp[i], V_tmp[0], &res_norm); CHKERRQ(ierr);
        ierr = VecDot(X_tmp[i], V_tmp[1], &evec_norm); CHKERRQ(ierr);
#if DEBUG_DETAIL
        ierr = PetscTime(&tvd2);CHKERRQ(ierr);
        Debug_Para->VDot_Time += (double)(tvd2-tvd1);
        Debug_Para->VDot_Iter += 2;
#endif
        //approx_eval[i] = res_norm/evec_norm;

#if DEBUG_DETAIL
        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
        ierr = VecAYPX(V_tmp[1], -approx_eval[i], V_tmp[0]); CHKERRQ(ierr);
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
        Debug_Para->AXPY_Time += (double)(tmv2-tmv1);
        Debug_Para->AXPY_Iter += 1;
#endif
        //||Au-\lambdaBu||/||Au||/\lambda
        //ierr = VecNorm(Vtmp[0], Norm_2, norm_2); CHKERRQ(ierr);
        //res  = norm_1/norm_2/eval[i];
        //Norm_2=1
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
        ierr = VecNorm(V_tmp[1], 1, &res_norm); CHKERRQ(ierr);
        ierr = VecNorm(X_tmp[i], 1, &evec_norm); CHKERRQ(ierr);
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
        Debug_Para->Norm_Time += (double)(tmv2-tmv1);
        Debug_Para->Norm_Iter += 2;
#endif
        res  = res_norm/evec_norm/approx_eval[i];
        RRes[i] = res;
	sum_res += res;
        if(i == 0)
        {
            max_res = res;
            min_res = res;
        }
        else
        {
            if(res > max_res)
                max_res = res;
            if(res < min_res)
                min_res = res;
        }
        if(res > abs_tol)
        {
            unlock[nunlocktmp] = i;
            nunlocktmp += 1;
        }
    }
    *nunlock = nunlocktmp;
    PetscPrintf(PETSC_COMM_WORLD,"nunlock(%d)= %d; max_res(%d)= %e; min_res(%d)= %e; sum_res(%d)=%e;\n", iter, nunlocktmp, iter, max_res, iter, min_res, iter, sum_res);
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    //printf("nunlock(%d)= %d; max_res(%d)= %e; min_res(%d)= %e, rank: %d\n", iter+1, nunlocktmp, iter+1, max_res, iter+1, min_res, rank);
    ierr = PetscOptionsGetInt(NULL, NULL, "-print_eval", &print_eval, NULL); CHKERRQ(ierr);
    if(print_eval == 1)
        PrintSmallEigen(iter, nev, approx_eval, NULL, NULL, 0, RRes);
    fflush(stdout);
    //free(Rayleigh);  Rayleigh = NULL;
#if DEBUG_TIME
    ierr = PetscTime(&t2);CHKERRQ(ierr);
    Debug_Para->Conv_Time += (double)(t2-t1);
    Debug_Para->One_Conv_Time = (double)(t2-t1);
#endif

    PetscFunctionReturn(0);
}

//获取d
PetscErrorCode GetPinV(PetscReal *AA, Vec *V, PetscInt dim_x, PetscInt last_dim_x, PetscInt *dim_xp, PetscInt dim_xpw, PetscInt nunlock, PetscInt *unlock, Vec *V_tmp, Vec *Orth_tmp, PetscInt *Ind, DEBUG_PARA *Debug_Para)
{
    PetscInt       i;
    PetscErrorCode ierr;
#if DEBUG_TIME
    PetscLogDouble t1, t2;
    ierr = PetscTime(&t1);CHKERRQ(ierr);
#endif
    //小规模正交化，构造d,构造AA_sub用于下次计算
    for( i=0; i<nunlock; i++ )
    {
        memset(AA+(dim_x+i)*dim_xpw, 0.0, dim_xpw*sizeof(PetscReal));
        memcpy(AA+(dim_x+i)*dim_xpw+last_dim_x, AA+unlock[i]*dim_xpw+last_dim_x, (dim_xpw-last_dim_x)*sizeof(PetscReal));
    }
    //小规模evec中，X部分是已经正交的（BB正交），所以对P部分正交化
    OrthogonalSmall(AA, NULL, dim_xpw, dim_x, dim_xp, Ind);
    //OrthogonalSmall(AA, NULL, dim_xpw, 0, dim_xp, Ind);
    //计算P所对应的长向量，存在V中
    ierr = GetRitzVectors(AA+dim_x*dim_xpw, V, V_tmp, dim_xpw, (*dim_xp)-dim_x, Debug_Para, 0);CHKERRQ(ierr);
    ChangeVecPointer(V+dim_x, V_tmp, Orth_tmp, (*dim_xp)-dim_x);
#if DEBUG_TIME
    ierr = PetscTime(&t2);CHKERRQ(ierr);
    Debug_Para->GetP_Time += (double)(t2-t1);
    Debug_Para->One_GetP_Time = (double)(t2-t1);
#endif
    //time[0] += (double)(t2-t1);
    PetscFunctionReturn(0);
}

void GetXinV(Vec *V, Vec *X_tmp, Vec *tmp, PetscInt dim_x)
{
    ChangeVecPointer(V, X_tmp, tmp, dim_x);
}

//如果对称，那么nl=0,LVecs=NULL;
void DenseVecsMatrixVecs(PetscReal *LVecs, PetscReal *DenseMat, PetscReal *RVecs, PetscReal *ProductMat, PetscInt nl, PetscInt nr, PetscInt dim, PetscReal *tmp)
{
    PetscInt  i, j;
    for( i=0; i<nr; i++ )
    {
        //t=A*u[i]
        DenseMatVec(DenseMat, RVecs+i*dim, tmp, dim);
        if(nl == 0)//对称
        {
            for( j=0; j<i+1; j++ )
            {
                ProductMat[i*nr+j] = VecDotVecSmall(RVecs+j*dim, tmp, dim);
                ProductMat[j*nr+i] = ProductMat[i*nr+j];
            }
        }
        else
        {
            for( j=0; j<nl; j++ )
            {
                ProductMat[i*nl+j] = VecDotVecSmall(LVecs+j*dim, tmp, dim);
            }
        }
    }
}

PetscErrorCode RayleighRitz(Mat A, Mat B, Vec *V, PetscReal *AA, PetscReal *approx_eval, PetscReal *AA_sub, PetscReal *AA_copy, PetscInt start, PetscInt last_dim, PetscInt dim, PetscInt dim_x, Vec tmp, PetscReal *small_tmp, DEBUG_PARA *Debug_Para)
{
    PetscErrorCode ierr;
#if DEBUG_TIME
    PetscLogDouble t1, t2;
    ierr = PetscTime(&t1);CHKERRQ(ierr);
#endif
    PetscReal      *work;
    PetscInt       info, lwork = 3*dim;
    work = (PetscReal*)calloc(lwork, sizeof(PetscReal));

#if A_PRODUCT
    ierr = GetLAPACKMatrix(B, V, AA, AA_sub, start, last_dim, dim, AA_copy, tmp, small_tmp, Debug_Para);CHKERRQ(ierr);
#else
    ierr = GetLAPACKMatrix(A, V, AA, AA_sub, start, last_dim, dim, AA_copy, tmp, small_tmp, Debug_Para);CHKERRQ(ierr);
#endif
#if 0
	ierr = MPI_Bcast(AA, dim*dim, MPI_DOUBLE, 0, PETSC_COMM_WORLD);CHKERRQ(ierr);
#endif
#if DEBUG_DETAIL
    PetscLogDouble tmv1, tmv2;
    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif

    LAPACKsyev_("V", "U", &dim, AA, &dim, approx_eval, work, &lwork, &info);

#if 0
    PetscMPIInt rank = 0;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    PetscInt i, j;
    for(i=0; i<dim; i++)
    {
      for(j=0; j<dim; j++)
      {
	printf("evec(%d, %d, %d) = %18.15lf\n;", i+1, j+1, rank, AA[i*dim+j]);
      }
    }
#endif
    //PetscPrintf(PETSC_COMM_WORLD, "A_PRODUCT: %d\n", A_PRODUCT);
#if A_PRODUCT
    SortEigen(AA, approx_eval, dim, dim_x);
#endif
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
    Debug_Para->Small_Time += (double)(tmv2-tmv1);
#endif

    free(work);  work = NULL;
#if DEBUG_TIME
    ierr = PetscTime(&t2);CHKERRQ(ierr);
    //time[0] += (double)(t2-t1);
    Debug_Para->Rayl_Time += (double)(t2-t1);
    Debug_Para->One_Rayl_Time = (double)(t2-t1);
#endif
    PetscFunctionReturn(0);
}

//用A内积的话，特征值从小到大，就是原问题特征值倒数的从小到大，所以顺序反向，同时特征值取倒数
void SortEigen(PetscReal *evec, PetscReal *eval, PetscInt dim, PetscInt dim_x)
{
    PetscInt head = 0, tail = dim-1;
    PetscReal *work = (PetscReal*)calloc(dim, sizeof(PetscReal));
	for( head=0; head<dim_x; head++ )
	{
	    tail = dim-1-head;
	    if(head < tail)
	    {
			memcpy(work, evec+head*dim, dim*sizeof(PetscReal));
			memcpy(evec+head*dim, evec+tail*dim, dim*sizeof(PetscReal));
			memcpy(evec+tail*dim, work, dim*sizeof(PetscReal));
			work[0] = eval[head];
			eval[head] = 1.0/eval[tail];
			eval[tail] = 1.0/work[0];
	    }
	    else
	    {
		    break;
	    }
	}
    free(work); work = NULL;
}

void Updatedim_x(PetscInt start, PetscInt end, PetscInt *dim_x, PetscReal *approx_eval)
{
    PetscInt tmp, i;
    tmp = start;
    //dsygv求出的特征值已经排序是ascending,从小到大
    //检查特征值的数值确定下次要进行计算的特征值个数
    for( i=start; i<end; i++ )
    {
        if((fabs(fabs(approx_eval[tmp]/approx_eval[tmp-1])-1))<0.2)
            tmp += 1;
        else
            break;
    }
    *dim_x = tmp;
}

PetscErrorCode VecsMatrixVecsForRayleighRitz(Mat A, Vec *V, PetscReal *AA, PetscInt start, PetscInt dim, Vec tmp, DEBUG_PARA *Debug_Para)
{
    PetscInt       i, j;
    PetscErrorCode ierr;
#if DEBUG_DETAIL
    PetscLogDouble tmv1, tmv2, tvd1, tvd2;
#endif
    for( i=start; i<dim; i++ )
    {
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
        ierr = MatMult(A, V[i], tmp); CHKERRQ(ierr);
#if DEBUG_DETAIL
        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
        Debug_Para->SPMV_Time += (double)(tmv2-tmv1);
        Debug_Para->SPMV_Iter += 1;
#endif
        for( j=0; j<i+1; j++ )
        {
#if DEBUG_DETAIL
            ierr = PetscTime(&tvd1);CHKERRQ(ierr);
#endif
            ierr = VecDot(V[j], tmp, AA+i*dim+j); CHKERRQ(ierr);
#if DEBUG_DETAIL
            ierr = PetscTime(&tvd2);CHKERRQ(ierr);
            Debug_Para->VDot_Time += (double)(tvd2-tvd1);
            Debug_Para->VDot_Iter += 1;
#endif
            AA[j*dim+i] = AA[i*dim+j];
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode VecsMatrixVecs(Vec *VL, Mat A, Vec *VR, PetscReal *AA, PetscInt nl, PetscInt nr, Vec tmp)
{
    PetscInt       i, j;
    PetscErrorCode ierr;
    for( i=0; i<nr; i++ )
    {
        ierr = MatMult(A, VR[i], tmp); CHKERRQ(ierr);
        for( j=0; j<nl; j++ )
        {
            ierr = VecDot(VL[j], tmp, AA+i*nl+j); CHKERRQ(ierr);
        }
    }
    PetscFunctionReturn(0);
}

//U = V*x
PetscErrorCode SumSeveralVecs(Vec *V, PetscReal *x, Vec U, PetscInt n_vec, DEBUG_PARA *Debug_Para)
{
    PetscInt       i;
    PetscErrorCode ierr;
#if DEBUG_DETAIL
    PetscLogDouble tmv1, tmv2;
    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
    ierr = VecAXPBYPCZ(U, x[0], x[1], 0.0, V[0], V[1]);
    for( i=2; i<n_vec; i++ )
    {
        ierr = VecAXPY(U, x[i], V[i]); CHKERRQ(ierr);
    }
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
    Debug_Para->AXPY_Time += (double)(tmv2-tmv1);
    Debug_Para->AXPY_Iter += n_vec;
#endif
    PetscFunctionReturn(0);
}

//EVEC = V*H_ev
PetscErrorCode GetRitzVectors(PetscReal *SmallEvec, Vec *V, Vec *RitzVec, PetscInt dim, PetscInt n_vec, DEBUG_PARA *Debug_Para, PetscInt if_time)
{
    PetscInt i;
    PetscErrorCode ierr;
#if DEBUG_TIME
    PetscLogDouble t1, t2;
    ierr = PetscTime(&t1);CHKERRQ(ierr);
#endif
    for( i=0; i<n_vec; i++ )
    {
        ierr = SumSeveralVecs(V, SmallEvec+i*dim, RitzVec[i], dim, Debug_Para);CHKERRQ(ierr);
    }
#if DEBUG_TIME
    ierr = PetscTime(&t2);CHKERRQ(ierr);
    Debug_Para->Ritz_Time += (double)(t2-t1);
    //Debug_Para->One_Ritz_Time = (double)(t2-t1);
    if(if_time == 1)
    {
        Debug_Para->GetX_Time += (double)(t2-t1);
        Debug_Para->One_GetX_Time = (double)(t2-t1);
    }
#endif
        //time[0] += (double)(t2-t1);
    PetscFunctionReturn(0);
}

void ChangeVecPointer(Vec *V_1, Vec *V_2, Vec *tmp, PetscInt size)
{
    memcpy(tmp, V_1, size*sizeof(Vec));
    memcpy(V_1, V_2, size*sizeof(Vec));
    memcpy(V_2, tmp, size*sizeof(Vec));
}
//对V的所有列向量做关于矩阵A的正交化，如果A=NULL，那么实际上做的是L2正交化
//V1:是一个临时的存储空间，是表示零向量的列的指针
//全部正交化，则start=0
PetscErrorCode GCG_Orthogonal(Vec *V, Mat B, PetscInt start, PetscInt *end, Vec *V_tmp, Vec *Nonzero_Vec, PetscInt *Ind, DEBUG_PARA *Debug_Para)
{
    PetscErrorCode ierr;
#if DEBUG_TIME
    PetscLogDouble t1, t2, tmv1, tmv2;
    ierr = PetscTime(&t1);CHKERRQ(ierr);
#endif
    PetscInt       i, j, n_nonzero = 0, n_zero = 0;//, reorth_time = 0;
    PetscReal      vin, vout, tmp, dd;
    //地址是int型，所以这里只要分配int空间就可以，不需要PetscReal**
    if(B == NULL)
    {
        for( i=start; i<(*end); i++ )
        {
            if(i == 0)
            {
                ierr = VecNorm(V[0], 1, &dd); CHKERRQ(ierr);
                if(dd > 10*ORTH_ZERO_TOL)
                {
                    ierr = VecScale(V[0], 1.0/dd); CHKERRQ(ierr);
                    Ind[0] = 0;
                    n_nonzero = 1;
                }
            }
            else
            {
                ierr = VecNorm(V[i], 1, &vout); CHKERRQ(ierr);
                do{
                    vin = vout;
                    for(j = 0; j < start; j++)
                    {
                        //计算 V[i]= V[i]-(V[i]^T*V[j])*V[j]
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
                        ierr = VecDot(V[i], V[j], &tmp); CHKERRQ(ierr);
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
                        Debug_Para->VDot_Time += (double)(tmv2-tmv1);
                        Debug_Para->VDot_Iter += 1;
#endif
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
                        ierr = VecAXPY(V[i], -tmp, V[j]); CHKERRQ(ierr);
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
                        Debug_Para->AXPY_Time += (double)(tmv2-tmv1);
                        Debug_Para->AXPY_Iter += 1;
#endif
                    }
                    for(j = 0; j < n_nonzero; j++)
                    {
                        //计算 V[i]= V[i]-(V[i]^T*V[Ind[j]])*V[Ind[j]]
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
                        ierr = VecDot(V[i], V[Ind[j]], &tmp); CHKERRQ(ierr);
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
                        Debug_Para->VDot_Time += (double)(tmv2-tmv1);
                        Debug_Para->VDot_Iter += 1;
#endif
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
                        ierr = VecAXPY(V[i], -tmp, V[Ind[j]]); CHKERRQ(ierr);
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
                        Debug_Para->AXPY_Time += (double)(tmv2-tmv1);
                        Debug_Para->AXPY_Iter += 1;
#endif
                    }
                    ierr = VecNorm(V[i], 1, &vout); CHKERRQ(ierr);
                }while(vout/vin < REORTH_TOL);
                if(vout > 10*ORTH_ZERO_TOL)
                {
                    ierr = VecScale(V[i], 1.0/vout); CHKERRQ(ierr);
                    Ind[n_nonzero++] = i;
                }//现在应该不需要free释放空间
                else
                {
                    //PetscPrintf(PETSC_COMM_WORLD, "'In GCG_Orthogonal, there is a zero vector! i = %d, start = %d, end: %d'\n", i, start, *end);
                    Nonzero_Vec[n_zero++] = V[i];
                }
            }
        }

    }
    else
    {
		for(i = 0; i < start; i++)
        {
#if DEBUG_DETAIL
            ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
			ierr = MatMult(B, V[i], V_tmp[i]); CHKERRQ(ierr);
#if DEBUG_DETAIL
            ierr = PetscTime(&tmv2);CHKERRQ(ierr);
            Debug_Para->SPMV_Time += (double)(tmv2-tmv1);
            Debug_Para->SPMV_Iter += 1;
#endif
        }
        for( i=start; i<(*end); i++ )
        {
            if(i == 0)
            {
                //计算 V[0]^T*A*V[0]
                dd = sqrt(VecMatrixVec(V[0], B, V[0], V_tmp[0], Debug_Para));
                if(dd > 10*ORTH_ZERO_TOL)
                {
                    ierr = VecScale(V[0], 1.0/dd); CHKERRQ(ierr);
                    Ind[n_nonzero++] = 0;
#if DEBUG_DETAIL
                    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
                    ierr = MatMult(B, V[0], V_tmp[0]); CHKERRQ(ierr);
#if DEBUG_DETAIL
                    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
                    Debug_Para->SPMV_Time += (double)(tmv2-tmv1);
                    Debug_Para->SPMV_Iter += 1;
#endif
                }
            }
            else
            {
                vout = sqrt(VecMatrixVec(V[i], B, V[i], V_tmp[start+n_nonzero], Debug_Para));
                //reorth_time = 0; 
                do{
                    vin = vout;
                    for(j = 0; j < start; j++)
                    {
                        //计算 V[i]= V[i]-(V[i]^T*V[j])_B*V[j]
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
                        ierr = VecDot(V[i], V_tmp[j], &tmp); CHKERRQ(ierr);
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
                        Debug_Para->VDot_Time += (double)(tmv2-tmv1);
                        Debug_Para->VDot_Iter += 1;
#endif
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
                        ierr = VecAXPY(V[i], -tmp, V[j]); CHKERRQ(ierr);
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
                        Debug_Para->AXPY_Time += (double)(tmv2-tmv1);
                        Debug_Para->AXPY_Iter += 1;
#endif
                    }
                    for(j = 0; j < n_nonzero; j++)
                    {
                        //计算 V[i]= V[i]-(V[i]^T*V[Ind[j]])_B*V[Ind[j]]
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
                        ierr = VecDot(V[i], V_tmp[start+j], &tmp); CHKERRQ(ierr);
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
                        Debug_Para->VDot_Time += (double)(tmv2-tmv1);
                        Debug_Para->VDot_Iter += 1;
#endif
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
                        ierr = VecAXPY(V[i], -tmp, V[Ind[j]]); CHKERRQ(ierr);
#if DEBUG_DETAIL
                        ierr = PetscTime(&tmv2);CHKERRQ(ierr);
                        Debug_Para->AXPY_Time += (double)(tmv2-tmv1);
                        Debug_Para->AXPY_Iter += 1;
#endif
                    }
                    vout = sqrt(VecMatrixVec(V[i], B, V[i], V_tmp[start+n_nonzero], Debug_Para));
                    //reorth_time += 1; 
                    //if((i>12)&&(i<23))
                    //    PetscPrintf(PETSC_COMM_WORLD, "P_orthogonal: %d, vout/vin: %e, vout: %e, EPS: %e\n", reorth_time, vout/vin, vout, EPS);
                }while(vout/vin < REORTH_TOL);
                if(vout > 10*ORTH_ZERO_TOL)
                {
                    ierr = VecScale(V[i], 1.0/vout); CHKERRQ(ierr);
#if DEBUG_DETAIL
                    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
                    ierr = MatMult(B, V[i], V_tmp[start+n_nonzero]); CHKERRQ(ierr);
#if DEBUG_DETAIL
                    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
                    Debug_Para->SPMV_Time += (double)(tmv2-tmv1);
                    Debug_Para->SPMV_Iter += 1;
#endif
                    Ind[n_nonzero++] = i;
                }
                else
                {
                    //PetscPrintf(PETSC_COMM_WORLD, "In GCG_Orthogonal, there is a zero vector! i = %d, start = %d, end: %d\n", i, start, *end);
                    Nonzero_Vec[n_zero++] = V[i];
                }
            }
        }
    }
    //接下来要把V的所有非零列向量存储在地址表格中靠前位置
    *end = start+n_nonzero;
    if(n_zero > 0)
    {
        for( i=0; i<n_nonzero; i++ )
            V[start+i] = V[Ind[i]];
        memcpy(V+(*end), Nonzero_Vec, n_zero*sizeof(Vec));
    }
#if DEBUG_TIME
    ierr = PetscTime(&t2);CHKERRQ(ierr);
    //time[0] += (double)(t2-t1);
    Debug_Para->Orth_Time += (double)(t2-t1);
    Debug_Para->One_Orth_Time = (double)(t2-t1);
#endif
    PetscFunctionReturn(0);
}

PetscReal VecMatrixVec(Vec a, Mat Matrix, Vec b, Vec temp, DEBUG_PARA *Debug_Para)
{
    PetscErrorCode ierr;
    PetscReal value=0.0;
#if DEBUG_DETAIL
    PetscLogDouble tmv1, tmv2;
    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
    ierr = MatMult(Matrix, b, temp); CHKERRQ(ierr);
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
    Debug_Para->SPMV_Time += (double)(tmv2-tmv1);
    Debug_Para->SPMV_Iter += 1;
#endif
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv1);CHKERRQ(ierr);
#endif
    ierr = VecDot(a, temp, &value); CHKERRQ(ierr);
#if DEBUG_DETAIL
    ierr = PetscTime(&tmv2);CHKERRQ(ierr);
    Debug_Para->VDot_Time += (double)(tmv2-tmv1);
    Debug_Para->VDot_Iter += 1;
#endif
    //value = IRA_VecDotVec(a, temp);
    return value;
}

//进行部分的正交化, 对V中start位置中后的向量与前start的向量做正交化，同时V的start之后的向量自己也做正交化
//dim_xpw表示V中总的向量个数
//V1:用来存零向量的地址指针
//对小规模的向量组V做正交化, B:表示度量矩阵,dim_xpw表示向量长度，dim_xp:向量个数，V_1：存储零向量的位置
//Vtmp:dim_xp*dim_xpw
void OrthogonalSmall(PetscReal *V, PetscReal **B, PetscInt dim_xpw, PetscInt dim_x, PetscInt *dim_xp, PetscInt *Ind)
{
    PetscInt i, j, n_nonzero = 0;//, reorth_time = 0;
    PetscReal vin, vout, tmp, dd;

    if(B == NULL)
    {
        for( i=dim_x; i<(*dim_xp); i++ )
        {
            if(i == 0)
            {
	        dd = NormVecSmall(V, dim_xpw);
                //ierr = VecNorm(V[0], 1, &dd); CHKERRQ(ierr);
                if(dd > 10*ORTH_ZERO_TOL)
                {
                    ScalVecSmall(1.0/dd, V, dim_xpw);
                    //Ind[n_nonzero++] = i;
                    //ierr = VecScale(V[0], 1.0/dd); CHKERRQ(ierr);
                    Ind[0] = 0;
                    n_nonzero = 1;
                }
            }
	    else
	    {

            vout = NormVecSmall(V+i*dim_xpw, dim_xpw);
            //reorth_time = 0; 
            do{
                vin = vout;
                for(j = 0; j < dim_x; j++)
                {
                    tmp = VecDotVecSmall(V+j*dim_xpw, V+i*dim_xpw, dim_xpw);
                    SmallAXPBY(-tmp, V+j*dim_xpw, 1.0, V+i*dim_xpw, dim_xpw);
                }
                for(j = 0; j < n_nonzero; j++)
                {
                    tmp = VecDotVecSmall(V+Ind[j]*dim_xpw, V+i*dim_xpw, dim_xpw);
                    SmallAXPBY(-tmp, V+Ind[j]*dim_xpw, 1.0, V+i*dim_xpw, dim_xpw);
                }
                vout = NormVecSmall(V+i*dim_xpw, dim_xpw);
                //reorth_time += 1; 
                //if((i>10)&&(i<(*dim_xp)))
                //    PetscPrintf(PETSC_COMM_WORLD, "small P_orthogonal: %d, vout/vin: %e, vout: %e, EPS: %e\n", reorth_time, vout/vin, vout, EPS);
            }while(vout/vin < REORTH_TOL);

            if(vout > 10*ORTH_ZERO_TOL)
            {
                ScalVecSmall(1.0/vout, V+i*dim_xpw, dim_xpw);
                Ind[n_nonzero++] = i;
            }
            else
            {
                //printf("in OrthogonalSmall, there appears a zero vector! i: %d\n", i);
                //PetscPrintf(PETSC_COMM_WORLD, "'in OrthogonalSmall, there appears a zero vector! i: %d'\n", i);
            }
	    }

        }
    }

    if(n_nonzero < (*dim_xp-dim_x))
    {
        *dim_xp = dim_x+n_nonzero;
        for( i=0; i<n_nonzero; i++ )
        {
            //printf("Ind[%d] = %d\n", i, Ind[i]);
            memcpy(V+(dim_x+i)*dim_xpw, V+Ind[i]*dim_xpw, dim_xpw*sizeof(PetscReal));
        }
    }
}

//右乘:b=Ax,A是方阵，按列优先存储
void DenseMatVec(PetscReal *DenseMat, PetscReal *x, PetscReal *b, PetscInt dim)
{
    PetscInt i;
    memset(b, 0.0, dim*sizeof(PetscReal));

    for( i=0; i<dim; i++ )
    {
        //for( j=0; j<dim; j++ )
        //    PetscPrintf(PETSC_COMM_WORLD, "b[%d, %d], need to zero: %e\n",i, 0, b[0]);
        SmallAXPBY(x[i], DenseMat+i*dim, 1.0, b, dim);
    }
}
//a=alpha*a, n:表示向量a的长度
void ScalVecSmall(PetscReal alpha, PetscReal *a, PetscInt n)
{
    PetscInt i;
    for( i=0; i<n; i++ )
        a[i] *= alpha;
}
//对计算向量a的范数，n：表示向量a的长度
PetscReal NormVecSmall(PetscReal *a, PetscInt n)
{
    PetscInt i;
    PetscReal value = 0.0;
    for( i=0; i<n; i++ )
        value += a[i]*a[i];
    return sqrt(value);
}
//计算向量a和b的内积，n：表示向量的长度
PetscReal VecDotVecSmall(PetscReal *a, PetscReal *b, PetscInt n)
{
    PetscInt i;
    PetscReal value = 0.0;
    for( i=0; i<n; i++ )
        value += a[i]*b[i];
    return value;
}

//b = alpha*a+beta*b,n表示向量的长度
void SmallAXPBY(PetscReal alpha, PetscReal *a, PetscReal beta, PetscReal *b, PetscInt n)
{
    PetscInt i;
    for( i=0; i<n; i++ )
    {
        //if( i== 0)
        //    PetscPrintf(PETSC_COMM_WORLD, "alpha: %e, a[0]: %e, beta: %e, b[0]: %e\n", alpha, a[0], beta, b[0]);
        b[i] = alpha*a[i]+beta*b[i];
    }
}

//PetscErrorCode LinearSolverCreate(KSP *ksp, Mat A, Mat T, PetscReal cg_tol, PetscInt nsmooth)
PetscErrorCode LinearSolverCreate(KSP *ksp, Mat A, Mat T, DEBUG_PARA *Debug_Para)
{
    PetscErrorCode ierr;
    PC pc;
    ierr = KSPCreate(PETSC_COMM_WORLD,ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(*ksp,A,T);CHKERRQ(ierr);
    /*
    ierr = KSPSetType(*ksp, KSPCG);CHKERRQ(ierr);
    ierr = KSPSetTolerances(*ksp, 0.1, PETSC_DEFAULT, PETSC_DEFAULT, Debug_Para->cg_max_it);CHKERRQ(ierr);
    ierr = KSPGetPC(*ksp, &pc);CHKERRQ(ierr);
    ierr = PCSetType(pc, PCHYPRE);CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc, "boomeramg");CHKERRQ(ierr);
    */
    ierr = KSPSetFromOptions(*ksp);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(*ksp, 1);
    KSPView(*ksp, PETSC_VIEWER_STDOUT_WORLD);
    PetscFunctionReturn(0);
}

PetscInt petscmax(PetscInt a, PetscInt b)
{
    if(a > b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

void CreateDEBUG_PARA(DEBUG_PARA **Debug_Para)
{
    (*Debug_Para) = (DEBUG_PARA*)malloc(sizeof(DEBUG_PARA));
    (*Debug_Para)->GetW_Time = 0.0;
    (*Debug_Para)->GetX_Time = 0.0;
    (*Debug_Para)->GetP_Time = 0.0;
    (*Debug_Para)->Ritz_Time = 0.0;
    (*Debug_Para)->Orth_Time = 0.0;
    (*Debug_Para)->Rayl_Time = 0.0;
    (*Debug_Para)->Conv_Time = 0.0;
    (*Debug_Para)->CG_Time   = 0.0;
    (*Debug_Para)->CG_Iter   = 0;
    (*Debug_Para)->SPMV_Time = 0.0;
    (*Debug_Para)->SPMV_Iter = 0;
    (*Debug_Para)->CGSPMV_Time = 0.0;
    (*Debug_Para)->CGSPMV_Iter = 0;
    (*Debug_Para)->VDot_Time = 0.0;
    (*Debug_Para)->VDot_Iter = 0;
    (*Debug_Para)->CGVDot_Time = 0.0;
    (*Debug_Para)->CGVDot_Iter = 0;
    (*Debug_Para)->Norm_Time = 0.0;
    (*Debug_Para)->Norm_Iter = 0;
    (*Debug_Para)->CGNorm_Time = 0.0;
    (*Debug_Para)->CGNorm_Iter = 0;
    (*Debug_Para)->AXPY_Time = 0.0;
    (*Debug_Para)->AXPY_Iter = 0;
    (*Debug_Para)->Small_Time = 0.0;
    (*Debug_Para)->cg_max_it = 1000;
    (*Debug_Para)->cg_rate   = 0.1;
}

void FreeDEBUG_PARA(DEBUG_PARA *Debug_Para)
{
    free(Debug_Para); Debug_Para = NULL;
}
