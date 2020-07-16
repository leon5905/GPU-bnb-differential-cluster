#building
mode=0
for i in "$*"
do
   case $i in
	debug)	mode=1
				break
				;;
	callgrind)	mode=2
						break
						;;
	gprof) mode=3
			break
			;;
   esac
done

echo $mode

if [ $mode = 0 ] 
then
	echo "Compiling Optmized Code"
nvcc -O3 -Xptxas -O3,-v --default-stream per-thread common.cpp kernel_trifle.cu trifle.cpp main.cpp -o exec.out\
    --generate-code arch=compute_52,code=sm_52 \
    --generate-code arch=compute_61,code=sm_61
elif [ $mode = 1 ] 
then
	#building -G device code, -g host code
	echo "Compiling Debug"
nvcc -G -g common.cpp kernel_trifle.cu trifle.cpp main.cpp -o exec.out\
    --generate-code arch=compute_52,code=sm_52 \
    --generate-code arch=compute_61,code=sm_61
elif [ $mode = 2 ] 
then
	echo "Compiling Callgrind (Performance)"
elif [ $mode = 3 ] 
then
	echo "Compiling Optmized Code with Gprof enabled"
nvcc -O3 -Xptxas -O3,-v --default-stream per-thread -pg common.cpp kernel_trifle.cu trifle.cpp main.cpp -o exec.out\
    --generate-code arch=compute_52,code=sm_52 \
    --generate-code arch=compute_61,code=sm_61
fi