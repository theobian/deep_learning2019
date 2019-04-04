## Requires 2 Arguments:
# 	Argument 1: number
#   	Argument 2: N ... number of digits the final number should contain
function generateNDigitNumber {
	local i orignumb2 orignumb N newnumb
	orignumb2=$1
	orignumb=$1
	N=$2
	newnumb=""

	# if length is negative or 0, then the original number is returned
	if (( N <= 0 )); then
		echo $1
	
	else
		# determine the number of digits the original number has
		n_digs=0
		while [[ ${orignumb2} -ne 0 ]];
		do
			let "orignumb2/=10"
			let "n_digs++"
			#orignumb2=((${orignumb2}/10))
			#n_digs=((${n_digs}+1))
	
		done
		let "n_zeros=$N-${n_digs}"
	
		for ((i=0;i<$n_zeros;i++))
		do
			newnumb=${newnumb}"0"
		done
		newnumb=${newnumb}${orignumb}
	
		echo ${newnumb}
	fi

}