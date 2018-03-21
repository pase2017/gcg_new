include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex1: ex1.o chkopts
	-${CLINKER} -o ../bin/ex1.exe ex1.o ${PETSC_KSP_LIB}
	${RM} ex1.o

ex_linear_withpre: ex_linear_withpre.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/ex_linear_withpre.exe ex_linear_withpre.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${PETSC_KSP_LIB}
	${RM} ex_linear_withpre.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

exgcg: exgcg.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/exgcg.exe exgcg.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${PETSC_KSP_LIB}
	${RM} exgcg.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

exgcg_petscbin_2: exgcg_petscbin_2.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/exgcg_petscbin_2.exe exgcg_petscbin_2.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${PETSC_KSP_LIB}
	${RM} exgcg_petscbin_2.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

exgcg_petscbin_1: exgcg_petscbin_1.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/exgcg_petscbin_1.exe exgcg_petscbin_1.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${PETSC_KSP_LIB}
	${RM} exgcg_petscbin_1.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

exgcg_petscbin: exgcg_petscbin.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/exgcg_petscbin.exe exgcg_petscbin.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${PETSC_KSP_LIB}
	${RM} exgcg_petscbin.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

extest: extest.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/extest.exe extest.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${PETSC_KSP_LIB}
	${RM} extest.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

exgcg_2A_new: exgcg_2A_new.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/exgcg_2A_new.exe exgcg_2A_new.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${PETSC_KSP_LIB}
	${RM} exgcg_2A_new.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

exgcg_2A: exgcg_2A.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/exgcg_2A.exe exgcg_2A.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${PETSC_KSP_LIB}
	${RM} exgcg_2A.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

exgcg_fd: exgcg_fd.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o chkopts
	-${CLINKER} -o ../bin/exgcg_fd.exe exgcg_fd.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o ${PETSC_KSP_LIB}
	${RM} exgcg_fd.o ReadWritePrint.o SlepcCG.o SlepcGCGEigen.o 

cleanall:
	rm -f *.exe ../bin/*.exe
