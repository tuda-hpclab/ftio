#! /bin/bash
# Contains all functions used

# Function to check if PORT is in use
function is_port_in_use() {
	local port_number=$1
	port_output=$(netstat -tlpn | grep ":$port_number ")
	if [[ ! -z "$port_output" ]]; then
		# Port is in use
		echo -e "${RED}Error: Port $port_number is already in use...${BLACK}"
		return 0 # true, port is in use
	else
		# Port is free
		echo -e "${BLUE}Port $port_number is available.${BLACK}"
		return 1 # false, port is free
	fi
}

function check_port() {

	# Check if PORT is available
	if is_port_in_use $PORT; then
		echo -e "${RED}Error: Port $PORT is already in use on ${ADDRESS_FTIO}. Terminating existing process...${BLACK}"

		# Use ss command for potentially more reliable process identification (uncomment)
		# process_id=$(ss -tlpn | grep :"$PORT " | awk '{print $NF}')

		# Use netstat if ss is unavailable (uncomment)
		# process_id=$(netstat -tlpn | grep :"$PORT " | awk '{print $7}' | cut -d'/' -f1)
		process_id=$(netstat -nlp | grep :"$PORT " | awk '{print $7}' | cut -d'/' -f1)

		if [[ ! -z "$process_id" ]]; then
			echo -e "${YELLOW}Terminating process with PID: $process_id${BLACK}"
			kill ${process_id} && echo -e "${GREEN}Using $PORT on ${ADDRESS_FTIO}.${BLACK}"
			return 0
		else
			echo -e "${RED}Failed to identify process ID for PORT $PORT.${BLACK}"
		fi
		exit 1
	else
		echo -e "${GREEN}Using $PORT on ${ADDRESS_FTIO}.${BLACK}"
	fi
}



# Function to update settings and manage configurations
function update_settings() {
    # Update job name and allocation call flags if dry run
    if [ "$DRY_RUN" = true ]; then
        local new_name="Dry_${JOB_ID}"
        # Assuming `alloc_call_flags` is a global variable or similar
        local alloc_call_flags="${ALLOC_CALL_FLAGS//${JOB_ID}/${new_name}}"
        JOB_ID=$new_name
    fi

    # Set absolute paths
    APP_DIR=$(eval echo "$APP_DIR")
    FTIO_BIN_LOCATION=$(eval echo "$FTIO_BIN_LOCATION")

    # Update exclude flags and log suffix
    if [ "$EXCLUDE_FTIO" = true ] && [ "$EXCLUDE_CARGO" = true ] && [ "$EXCLUDE_DEMON" = true ] && [ "$EXCLUDE_PROXY" = true ]; then
        EXCLUDE_ALL=true
    fi

    if [ "$EXCLUDE_ALL" = true ]; then
        EXCLUDE_FTIO=true
        EXCLUDE_CARGO=true
        EXCLUDE_DEMON=true
        EXCLUDE_PROXY=true
    fi

    if [ "$CLUSTER" = false ] && [ "$NODES" -gt 1 ]; then
        PROCS=$NODES
        NODES=1
        echo -e "${JIT}${GREEN} >> correcting nodes to ${NODES} and processes to ${PROCS} ${BLACK}"
    fi

    LOG_SUFFIX="DPCF"
    if [ "$EXCLUDE_DEMON" = true ]; then
        PROCS_DEMON=0
        LOG_SUFFIX=${LOG_SUFFIX//D/}
    fi
    if [ "$EXCLUDE_PROXY" = true ]; then
        PROCS_PROXY=0
        LOG_SUFFIX=${LOG_SUFFIX//P/}
    fi
    if [ "$EXCLUDE_CARGO" = true ]; then
        PROCS_CARGO=0
        LOG_SUFFIX=${LOG_SUFFIX//C/}
    fi
    if [ "$EXCLUDE_FTIO" = true ]; then
        PROCS_FTIO=0
        LOG_SUFFIX=${LOG_SUFFIX//F/}
    fi

    if [ "$SET_TASKS_AFFINITY" = true ]; then
        TASK_SET_0="taskset -c 0-$((PROCS / 2 - 1))"
        if [ $((PROCS - PROCS / 2)) -ge $PROCS_APP ]; then
            TASK_SET_1="taskset -c $((PROCS / 2))-$((PROCS - 1))"
        fi
    fi


    # Output updated settings for verification (optional)
    echo -e "${JIT}${GREEN} >> Updated settings:${BLACK}"
    echo -e "${JIT}${GREEN} >> Job Name: $JOB_ID ${BLACK}"
    echo -e "${JIT}${GREEN} >> Allocation Call Flags: $alloc_call_flags ${BLACK}"
    echo -e "${JIT}${GREEN} >> App Directory: $APP_DIR ${BLACK}"
    echo -e "${JIT}${GREEN} >> FTIO Bin Location: $FTIO_BIN_LOCATION ${BLACK}"
    echo -e "${JIT}${GREEN} >> Procs: $PROCS ${BLACK}"
    echo -e "${JIT}${GREEN} >> Procs Demon: $PROCS_DEMON ${BLACK}"
    echo -e "${JIT}${GREEN} >> Procs Proxy: $PROCS_PROXY ${BLACK}"
    echo -e "${JIT}${GREEN} >> Procs Cargo: $PROCS_CARGO ${BLACK}"
    echo -e "${JIT}${GREEN} >> Procs FTIO: $PROCS_FTIO ${BLACK}"
    echo -e "${JIT}${GREEN} >> Procs App: $PROCS_APP ${BLACK}"
    echo -e "${JIT}${GREEN} >> Log Suffix: $LOG_SUFFIX ${BLACK}"
    if [ "$SET_TASKS_AFFINITY" = true ]; then
        echo -e "${JIT}${GREEN} >> Task Set 0: $TASK_SET_0 ${BLACK}"
        echo -e "${JIT}${GREEN} >> Task Set 1: $TASK_SET_1 ${BLACK}"
    fi
}

# Function to allocate resources
function allocate() {
    APP_NODES=1

    if [ "$CLUSTER" == true ]; then
        if [ "$JOB_ID" -eq 0 ]; then
            # Allocating resources
            echo -e "\n${JIT}${GREEN} ####### Allocating resources${BLACK}"

            local call="salloc -N ${NODES} -t ${MAX_TIME} ${ALLOC_CALL_FLAGS}"

            echo -e "${JIT}${CYAN} >> Executing: ${call} ${BLACK}"
            # Execute the salloc command
            eval "$call"

            # Get JOB_ID
            JOB_ID=$(squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" | grep "${JOB_NAME}" | sort -k1,1rn | head -n 1 | awk '{print $1}')

            if [ -z "$JOB_ID" ]; then
                JOB_ID=0
            fi
        else
            echo -e "${JIT}${GREEN} ## Using allocation with id: ${JOB_ID} ${BLACK}"
        fi

        if [ "$JOB_ID" -ne 0 ]; then
            # Get NODES_ARR
            NODES_ARR=($(scontrol show hostname $(squeue -j ${JOB_ID} -o "%N" | tail -n +2)))

            # Write to hostfile_mpi
            printf "%s\n" "${NODES_ARR[@]}" > ${APP_DIR}/hostfile_mpi

            if [ "${#NODES_ARR[@]}" -gt 0 ]; then
                # Get FTIO node
                FTIO_NODE="${NODES_ARR[-1]}"
                SINGLE_NODE="${NODES_ARR[0]}"

                if [ "${#NODES_ARR[@]}" -gt 1 ]; then
                    FTIO_NODE_COMMAND="--nodelist=${FTIO_NODE}"
                    APP_NODES_COMMAND="--nodelist=$(echo ${NODES_ARR[@]/${FTIO_NODE}/} | tr ' ' ',')"
                    SINGLE_NODE_COMMAND="--nodelist=${SINGLE_NODE}"

                    APP_NODES=$((${#NODES_ARR[@]} - 1))

                    # Remove FTIO node from hostfile_mpi
                    sed -i "/${FTIO_NODE}/d" ${APP_DIR}/hostfile_mpi
                fi

                echo -e "${JIT}${GREEN} >> JIT Job Id: ${JOB_ID} ${BLACK}"
                echo -e "${JIT}${GREEN} >> Allocated Nodes: ${#NODES_ARR[@]} ${BLACK}"
                echo -e "${JIT}${GREEN} >> FTIO Node: ${FTIO_NODE} ${BLACK}"
                echo -e "${JIT}${GREEN} >> APP Node command: ${APP_NODES_COMMAND} ${BLACK}"
                echo -e "${JIT}${GREEN} >> FTIO Node command: ${FTIO_NODE_COMMAND} ${BLACK}"
                echo -e "${JIT}${CYAN} >> content of ${APP_DIR}/hostfile_mpi: \n$(cat ${APP_DIR}/hostfile_mpi) ${BLACK}\n"
            else
                echo -e "${JIT}${RED} >> JOB_ID could not be retrieved ${BLACK}"
            fi
        else
            APP_NODES=${NODES}
        fi
    else
        APP_NODES=${NODES}
    fi
}

# Start FTIO
function start_ftio() {

	if [ "$EXCLUDE_FTIO" == true ] || [ "${EXCLUDE_ALL}" == true ]; then
		echo -e "\n${JIT}${YELLOW} ####### Skipping FTIO ${BLACK}"
		# Set relevant files using regex
		relevant_files true

		if [ "${EXCLUDE_CARGO}" == false ]; then
			# echo -e "${JIT}${YELLOW} Executing the calls bellow used later for staging out${BLACK}"
			call_0="srun --jobid=${JOB_ID} ${SINGLE_NODE_COMMAND} --disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 ${CARGO_CLI}/cargo_ftio --server ${CARGO_SERVER} --run"
			call_1="srun --jobid=${JOB_ID} ${SINGLE_NODE_COMMAND} --disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 ${CARGO_CLI}/ccp --server ${CARGO_SERVER} --input / --output ${STAGE_OUT_PATH} --if gekkofs --of parallel"
			execute "${call_0}"
			execute "${call_1}"
		fi
	else
		echo -e "\n${JIT}${GREEN} ####### Starting FTIO ${BLACK}"
		# Set relevant files using regex
		relevant_files true
		# set -x
		if [ "$CLUSTER" == true ]; then
			source ${FTIO_ACTIVATE}

			# One node is only for FTIO
			echo -e "${JIT}${CYAN} >> FTIO started on node ${FTIO_NODE}, remainng nodes for the application: ${APP_NODES} each with ${PROCS} processes ${BLACK}"
			echo -e "${JIT}${CYAN} >> FTIO is listening node is ${ADDRESS_FTIO}:${PORT} ${BLACK}"

			# call
			local call="srun --jobid=${JOB_ID} ${FTIO_NODE_COMMAND} --disable-status -N 1 --ntasks=1 --cpus-per-task=${PROCS} --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 predictor_jit  --zmq_address ${ADDRESS_FTIO} --zmq_port ${PORT} --cargo_cli ${CARGO_CLI} --cargo_server ${CARGO_SERVER} --cargo_out ${STAGE_OUT_PATH} "
		else
			#clean port
			check_port
			local call="predictor_jit  --zmq_address ${ADDRESS_FTIO} --zmq_port ${PORT} --cargo_cli ${CARGO_CLI} --cargo_server ${CARGO_SERVER} --cargo_out ${STAGE_OUT_PATH} "
			# 2>&1 | tee  ./ftio_${NODES}.txt
		fi
		# set -o xtrace
		# execute "${call}"
		execute "${call} 2> >(tee -a ${FTIO_ERR} >&2) | tee -a ${FTIO_LOG}"
		echo -e "\n\n"
	fi
}

# Start the Server
function start_geko_demon() {
	if [ "${EXCLUDE_DEMON}" == true ]; then
		echo -e "\n${JIT}${YELLOW} ####### Skipping GKFS DEMON ${BLACK}"
	else
		echo -e "\n${JIT}${GREEN} ####### Starting GKFS DEOMON ${BLACK}"
		# creat host file
		create_hostfile
		# set -x
		if [ "$CLUSTER" == true ]; then
			# Display Demon
			call_0="srun --jobid=${JOB_ID} ${SINGLE_NODE_COMMAND} -N 1 --ntasks=1 mkdir -p ${GKFS_MNTDIR}"
			# EXCLUDE_PROXY=false

			if [ "$EXCLUDE_PROXY" == true ]; then
				# Demon call
				local call="srun --export=ALL,GKFS_DAEMON_LOG_LEVEL=trace --jobid=${JOB_ID} ${APP_NODES_COMMAND} --disable-status -N ${APP_NODES} --ntasks=${APP_NODES} --cpus-per-task=${PROCS_DEMON} --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 ${GKFS_DEMON}  -r ${GKFS_ROOTDIR} -m ${GKFS_MNTDIR} -H ${GKFS_HOSTFILE}  -c -l ib0 -P ofi+sockets "
			else
				# Demon call no proxy
				local call="srun --export=ALL,GKFS_DAEMON_LOG_LEVEL=trace --jobid=${JOB_ID} ${APP_NODES_COMMAND} --disable-status -N ${APP_NODES} --ntasks=${APP_NODES} --cpus-per-task=${PROCS_DEMON} --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 ${GKFS_DEMON}  -r ${GKFS_ROOTDIR} -m ${GKFS_MNTDIR} -H ${GKFS_HOSTFILE}  -c -l ib0 -P ofi+sockets -p ofi+verbs -L ib0"
			fi

		else
			call_0="mkdir -p ${GKFS_MNTDIR}"

			# Geko Demon call
			# local call="GKFS_DAEMON_LOG_LEVEL=info ${GKFS_DEMON} -r ${GKFS_ROOTDIR} -m ${GKFS_MNTDIR} -H ${GKFS_HOSTFILE} -p ofi+tcp"
			local call="GKFS_DAEMON_LOG_LEVEL=info ${GKFS_DEMON} -r ${GKFS_ROOTDIR} -m ${GKFS_MNTDIR} -H ${GKFS_HOSTFILE} -c -l lo -P ofi+tcp --proxy-listen lo --proxy-protocol ofi+tcp"
			#-c --auto-sm
		fi
		echo -e "${JIT}${CYAN} >> Creating Directory ${BLACK}"
		execute "${call_0}"
		echo -e "${JIT}${CYAN} >> Starting Demons ${BLACK}"
		execute "${call}"
		echo -e "\n\n"
	fi
}

function start_geko_proxy() {

	if [ "${EXCLUDE_PROXY}" == true ]; then
		echo -e "\n${JIT}${YELLOW} ####### Skipping GKFS PROXY ${BLACK}"
	else
		echo -e "\n${JIT}${GREEN} ####### Starting GKFS PROXY ${BLACK}"
		if [ "$CLUSTER" == true ]; then
			# Proxy call
			local call="srun --jobid=${JOB_ID} ${APP_NODES_COMMAND} --disable-status -N ${APP_NODES} --ntasks=${APP_NODES} --cpus-per-task=${PROCS_PROXY} --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 ${GKFS_PROXY}  -H ${GKFS_HOSTFILE} -P ${GKFS_PROXYFILE} -p ofi+verbs"
			#
			# old
			# srun --jobid=${JOB_ID} ${APP_NODES_COMMAND} --disable-status -N ${APP_NODES} --ntasks=${APP_NODES} --cpus-per-task=${PROCS} \
			# --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 \
			# ${GKFS_PROXY}  \
			# -H ${GKFS_HOSTFILE} -p ofi+verbs -P ${GKFS_PROXYFILE}
		else

			local call="${GKFS_PROXY} -H ${GKFS_HOSTFILE} -p ofi+tcp -P ${GKFS_PROXYFILE}"
		fi
		execute "${call} 2> >(tee -a ${GEKKO_PROXY_LOG} >&2) | tee -a ${GEKKO_PROXY_ERR}"
		echo -e "\n\n"
	fi
}

function start_cargo() {
	if [ "${EXCLUDE_CARGO}" == true ]; then
		echo -e "\n${JIT}${YELLOW} ####### Skipping Cargo ${BLACK}"
	else
		echo -e "\n${JIT}${GREEN} ####### Starting Cargo ${BLACK}"
		# set -x
		if [ "$CLUSTER" == true ]; then

			# One instance per node
			# local call="srun --export=LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE},LIBGKFS_PROXY_PID_FILE=${GKFS_PROXYFILE},LD_LIBRARY_PATH=${LD_LIBRARY_PATH} --jobid=${JOB_ID} ${APP_NODES_COMMAND} --disable-status -N ${APP_NODES} --ntasks=$((APP_NODES * PROCS_CARGO)) --cpus-per-task=${PROCS_CARGO} --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0  ${CARGO} --listen ofi+sockets://ib0:62000 -b 65536"

			local call="srun --export=LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE},LIBGKFS_PROXY_PID_FILE=${GKFS_PROXYFILE},LD_LIBRARY_PATH=${LD_LIBRARY_PATH} --jobid=${JOB_ID} ${APP_NODES_COMMAND} --disable-status -N ${APP_NODES} --ntasks=${APP_NODES} --cpus-per-task=${PROCS_CARGO} --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0  ${CARGO} --listen ofi+sockets://ib0:62000 -b 65536"

		else
			# local call="mpiexec -np 4 --oversubscribe -x LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH} taskset -c 11-14  ${CARGO} --listen ofi+tcp://127.0.0.1:62000 -b 65536 "
			local call="mpiexec -np 2 --oversubscribe -x LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH} ${CARGO} --listen ofi+tcp://127.0.0.1:62000 -b 65536 "
		fi
		execute "${call} 2> >(tee -a ${CARGO_ERR} >&2) | tee -a ${CARGO_LOG}"
		echo -e "\n\n"
	fi
}

# Application call
function start_application() {
	echo -e "\n${JIT}${GREEN} ####### Executing Application ${BLACK}"
	original_dir=$(pwd)
	echo ">> Current directory ${original_dir}"

	# Change to the application directory
	cd "${APP_DIR}"
	echo ">> Changing directory to $(pwd)"

	if [ "$DRY_RUN" != true ]; then
		check_setup
	fi

	if [ "$CLUSTER" == true ]; then

		# without FTIO
		#? [--stag in (si)--]               [--stag out (so)--]
		#?              [---APP---]
		# with FTIO
		#? [--stag in--]   [so]  [so] ... [so]
		#?              [---APP---]
		# # app call
		# if [ "${EXCLUDE_ALL}" == true ]; then
		# 	# run without jit tools
		# 	local call="${PRECALL} mpiexec -np ${PROCS} --oversubscribe --hostfile ${APP_DIR}/hostfile_mpi --map-by node ${APP_CALL}"
		# elif [ "${EXCLUDE_FTIO}" == true ]; then
		# 	# run without FTIO
		# 	local call="${PRECALL} mpiexec -np ${PROCS} --oversubscribe --hostfile ${APP_DIR}/hostfile_mpi --map-by node -x LIBGKFS_LOG=errors -x LD_PRELOAD=${GKFS_INTERCEPT} -x LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} -x LIBGKFS_PROXY_PID_FILE=${GKFS_PROXYFILE} ${APP_CALL}"
		# else
		# 	# run with jit tools
		# 	local call="${PRECALL} mpiexec -np ${PROCS} --oversubscribe --hostfile ${APP_DIR}/hostfile_mpi --map-by node -x LIBGKFS_LOG=errors -x LIBGKFS_ENABLE_METRICS=on -x LIBGKFS_METRICS_IP_PORT=${ADDRESS_FTIO}:${PORT} -x LD_PRELOAD=${GKFS_INTERCEPT} -x LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} -x LIBGKFS_PROXY_PID_FILE=${GKFS_PROXYFILE} ${APP_CALL}"
		# fi
		additional_arguments=""
		# Check if FTIO is not excluded
		if [ "$EXCLUDE_FTIO" != true ]; then
			additional_arguments+="-x LIBGKFS_METRICS_IP_PORT=${ADDRESS_FTIO}:${PORT} -x LIBGKFS_ENABLE_METRICS=on "
		fi

		# Check if Proxy is not excluded
		if [ "$EXCLUDE_PROXY" != true ]; then
			additional_arguments+="-x LIBGKFS_PROXY_PID_FILE=${GKFS_PROXYFILE} "
		fi

		# Check if Demon is not excluded
		if [ "$EXCLUDE_DEMON" != true ]; then
			additional_arguments+="-x LIBGKFS_LOG=info,warnings,errors "
			additional_arguments+="-x LIBGKFS_LOG_OUTPUT=${GEKKO_CLIENT_LOG} "
			additional_arguments+="-x LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} "
			additional_arguments+="-x LD_PRELOAD=${GKFS_INTERCEPT} "
		fi

		# Construct the mpiexec call command
		call="mpiexec -np $((APP_NODES * PROCS_APP)) --oversubscribe "
		call+="--hostfile ${APP_DIR}/hostfile_mpi --map-by node "
		call+="${additional_arguments} "
		call+="${TASK_SET_1} ${APP_CALL}"


	else
		# app call
		if [ "${EXCLUDE_ALL}" == true ]; then
			local call=" mpiexec -np ${PROCS} --oversubscribe ${APP_CALL}"
		elif [ "${EXCLUDE_FTIO}" == true ]; then
			local call=" mpiexec -np ${PROCS} --oversubscribe -x LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} -x LIBGKFS_LOG=none -x LIBGKFS_PROXY_PID_FILE=${GKFS_PROXYFILE} -x LD_PRELOAD=${GKFS_INTERCEPT} ${APP_CALL}"
		else
			local call=" mpiexec -np ${PROCS} --oversubscribe -x LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} -x LIBGKFS_LOG=none -x LIBGKFS_ENABLE_METRICS=on -x LIBGKFS_METRICS_IP_PORT=${ADDRESS_FTIO}:${PORT} -x LIBGKFS_PROXY_PID_FILE=${GKFS_PROXYFILE} -x LD_PRELOAD=${GKFS_INTERCEPT} ${APP_CALL}"
		fi
	fi

	start=$(date +%s.%N)
	execute "${call} 2> >(tee -a ${APP_ERR} >&2) | tee -a ${APP_LOG}"
	end=$(date +%s.%N)
	FINISH=true

	elapsed_time "Application finished" ${start} ${end}
	APP_TIME=$(awk "BEGIN {print $end - $start}")

	
	cd "${original_dir}"
	echo ">> Changing directory to $(pwd)"
}

function stage_out() {

	if [ "${EXCLUDE_ALL}" == true ]; then
		echo -e "\n${JIT}${YELLOW} ####### Skipping Stage out ${BLACK}"
	else
		echo -e "\n${JIT}${YELLOW} ####### Stagin out ${BLACK}"

		if [ "$EXCLUDE_CARGO" == true ]; then
				local additional_arguments=$(generate_additional_arguments)
				local call="${additional_arguments} cp -r ${GKFS_MNTDIR}/* ${STAGE_OUT_PATH}"
				start=$(date +%s.%N)
				execute "${call}" 
				end=$(date +%s.%N)
				elapsed_time "Stage out" ${start} ${end}
				STAGE_OUT_TIME=$(awk "BEGIN {print $end - $start}")
		else

			local files=$(LD_PRELOAD=${GKFS_INTERCEPT} LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} ls ${GKFS_MNTDIR})
			echo -e "${JIT}${CYAN} >> geko_ls ${GKFS_MNTDIR}: \n${files} ${BLACK}\n"

			reset_relevant_files

			# stage out call on any compute node
			if [ "$CLUSTER" == true ]; then
				local call="srun --jobid=${JOB_ID} ${SINGLE_NODE_COMMAND} --disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 ${CARGO_CLI}/cargo_ftio --server ${CARGO_SERVER} --run"
			else
				local call="mpiexec -np 1 --oversubscribe ${CARGO_CLI}/cargo_ftio --server ${CARGO_SERVER} --run"
			fi

			start=$(date +%s.%N)
			execute_and_wait_stage_out "${call}" ${CYAN} " > Stage out: ${call}" "Transfer finished for"
			end=$(date +%s.%N)
			elapsed_time "Stage out" ${start} ${end}
			STAGE_OUT_TIME=$(awk "BEGIN {print $end - $start}")
		fi
		# set ignored files to default again
		relevant_files
	fi
}

function stage_in() {
	if [ "${EXCLUDE_ALL}" == true ]; then
		echo -e "\n${JIT}${YELLOW} ####### Skipping Stage in ${BLACK}"
	else
		echo -e "\n${JIT}${YELLOW} ####### Stagin in ${BLACK}"

		if [ "$EXCLUDE_CARGO" == true ]; then
				local additional_arguments=$(generate_additional_arguments)
				local call="${additional_arguments} cp -r ${STAGE_IN_PATH}/* ${GKFS_MNTDIR}"
				start=$(date +%s.%N)
				execute "${call}" 
				end=$(date +%s.%N)
				elapsed_time "Stage out" ${start} ${end}
				STAGE_OUT_TIME=$(awk "BEGIN {print $end_time - $start_time}")
		else
			# stage in call on any compute node
			if [ "$CLUSTER" == true ]; then

				local call="srun --jobid=${JOB_ID} ${SINGLE_NODE_COMMAND} --disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 ${CARGO_CLI}/ccp --server ${CARGO_SERVER} --output / --input ${STAGE_IN_PATH} --of gekkofs --if parallel"
				# local call="LD_PRELOAD=${GKFS_INTERCEPT}  LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE}  cp -r ${STAGE_IN_PATH}/* ${GKFS_MNTDIR}"
				# local call="srun --export=LD_LIBRARY_PATH=${LD_LIBRARY_PATH},LD_PRELOAD=${GKFS_INTERCEPT},LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} --jobid=${JOB_ID} ${SINGLE_NODE_COMMAND} --disable-status -N 1 --ntasks=1 /usr/bin/cp -r ${STAGE_IN_PATH}/* ${GKFS_MNTDIR}"
			else
				local call="mpiexec -np 1 --oversubscribe ${CARGO_CLI}/ccp --server ${CARGO_SERVER} --output / --input ${STAGE_IN_PATH}  --of gekkofs --if parallel"
			fi
		
			start=$(date +%s.%N)
			execute_and_wait_msg "${call}" ${CYAN} " > Stage in: ${call}" "retval: CARGO_SUCCESS, status: {state: completed"
			end=$(date +%s.%N)
			elapsed_time "Stage in" ${start} ${end}
			STAGE_IN_TIME=$(awk "BEGIN {print $end_time - $start_time}")

			local files=$(LD_PRELOAD=${GKFS_INTERCEPT} LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} ls ${GKFS_MNTDIR})
			echo -e "${JIT}${CYAN} >> geko_ls ${GKFS_MNTDIR}:${BLACK} \n${files} ${BLACK}\n"
		fi
	fi
}

#precall
function pre_call() {
	if [ -n "${PRE_APP_CALL}" ]; then
		echo -e "\n${JIT}${GREEN} ####### Pre-application Call ${BLACK}"
		execute "${PRE_APP_CALL} 2> >(tee -a ${FTIO_ERR} >&2) | tee -a ${FTIO_LOG}" 
	fi
}

# post call
function post_call() {
	if [ -n "${POST_APP_CALL}" ]; then
		echo -e "\n${JIT}${GREEN} ####### Post-application Call ${BLACK}"
		execute "${POST_APP_CALL} 2> >(tee -a ${FTIO_ERR} >&2) | tee -a ${FTIO_LOG}"  
	fi
}

function soft_kill() {
	echo -e "\n${JIT}${GREEN} ####### Soft kill ${BLACK}"

	if [ "$EXCLUDE_FTIO" == false ]; then
		shut_down "FTIO" ${FTIO_PID} &
		echo -e "${JIT}${CYAN} >> killed FTIO ${BLACK}"
	fi
	if [ "$EXCLUDE_DEMON" == false ]; then
		shut_down "GEKKO" ${GEKKO_DEMON_PID} &
		echo -e "${JIT}${CYAN} >> killed GEKKO DEMON ${BLACK}"
	fi
	if [ "$EXCLUDE_PROXY" == false ]; then
		shut_down "GEKKO" ${GEKKO_PROXY_PID} &
		echo -e "${JIT}${CYAN} >> killed GEKKO PROXY ${BLACK}"
	fi
	if [ "$EXCLUDE_CARGO" == false ]; then
		shut_down "CARGO" ${CARGO_PID} &
		echo -e "${JIT}${CYAN} >> killed CARGO ${BLACK}"
	fi
}

function hard_kill() {
	if [ "$CLUSTER" == true ] && [ "$STATIC_ALLOCATION" = false ]; then
		echo -e "\n${JIT}${GREEN} ####### Hard kill ${BLACK}"
		scancel ${JOB_ID} || true
	else
		echo -e "\n${JIT}${GREEN} ####### Hard kill ${BLACK}"
		kill $(ps -aux | grep ${GKFS_DEMON} | grep -v grep | awk '{print $2}') || true
		kill $(ps -aux | grep ${GKFS_PROXY} | grep -v grep | awk '{print $2}') || true
		kill $(ps -aux | grep ${CARGO} | grep -v grep | awk '{print $2}') || true
		kill $(ps -aux | grep "$(dirname "$FTIO_ACTIVATE")/predictor_jit" | grep -v grep | awk '{print $2}') || true
	fi
}
# Function to handle SIGINT (Ctrl+C)
function handle_sigint() {
	echo "${JIT} > Keyboard interrupt detected. Exiting script."
	soft_kill
	scancle ${JOB_ID}
	exit 0
}

function check_finish() {
	# Set trap to handle SIGINT
	trap 'handle_sigint' SIGINT

	while :; do
		if [ "$FINISH" == true ]; then
			echo "${JIT} > FINISH flag is true. Exiting script in 10 sec."
			sleep 10
			exit 0
		fi
	done
}


# Function to show error usage
error_usage() {
    echo -e "
Usage: $0 [OPTION]...

    -a | --address: X.X.X.X <string>
        default: ${YELLOW}${ADDRESS_FTIO}${BLACK}
        Address where FTIO is executed. On a cluster, this is found 
        automatically by determining the address of the node where FTIO 
        runs.

    -r | --port: XXXX <int>
        default: ${YELLOW}${PORT}${BLACK}
        Port for FTIO and GekkoFS.

    -n | --nodes: X <int>
        default: ${YELLOW}${NODES}${BLACK}
        Number of nodes to run the setup. In cluster mode, FTIO is 
        executed on a single node, while the rest (including the
        application) get X-1 nodes.

    -t | --max-time: X <int>
        default: ${YELLOW}${MAX_TIME}${BLACK}
        Max time for the execution of the setup in minutes.

    -j | --job-id: X <int>
        default: ${YELLOW}Auto detected${BLACK}
        Skips allocating new resources and uses the job ID.

    -l | --log-name: <str>
        default: ${YELLOW}Autoset to number of nodes and job ID${BLACK}
        If provided, sets the name of the directory where the logs are stored.

    -i | --install-location: full_path <str>
        default: ${YELLOW}${INSTALL_LOCATION}${BLACK}
        Installs everything in the provided directory.

    -o | --omp-threads: X <int>
        default: ${YELLOW}${OMP_THREADS}${BLACK}
        OpenMP threads used.
	
	-c | --total_procs: X <int>
        default: ${YELLOW}${PROCS}${BLACK}
        Total procs on node

    -p | --procs-list: x,x,..,x <list>
        default: ${YELLOW}${PROCS_APP},${PROCS_DEMON},${PROCS_PROXY},${PROCS_CARGO},${PROCS_FTIO}${BLACK}
        List of tasks per node/cpu per proc for app, demon, proxy, cargo, 
        and ftio, respectively. Assignment is from right to left depending 
        on the length of the list.

    -e | --exclude: <str>,<str>,...,<str>
        default: ${YELLOW}ftio${BLACK}
        If this flag is provided, the setup is executed without the tool(s).
        Supported options include: ftio, demon, proxy, gkfs (demon + proxy), 
        cargo, and all (same as -x).

    -x | --exclude-all
        default: ${YELLOW}${EXCLUDE_ALL}${BLACK}
        If this flag is provided, the setup is executed without FTIO, 
        GekkoFS, and Cargo.

    -d | --dry-run
        default: ${YELLOW}${DRY_RUN}${BLACK}
        If provided, the tools and the app are not executed.

    -v | --verbose
        default: ${YELLOW}${VERBOSE}${BLACK}
        If provided, the tools output of each step is shown.

    -y | --skip-confirm
        default: ${YELLOW}${SKIP_CONFIRM}${BLACK}
        If this flag is provided, the setup automatically cancels running jobs 
        named JIT.

    -h | --help
        Show this help message.

\n---- exit ----
    "
}


abort() {
	echo >&2 '
***************
*** ABORTED ***
***************
'
	echo "An error occurred. Exiting..." >&2
	exit 1
}

function install_all() {
	#create dir
	trap 'abort' 0
	set -e #exit on fail

	echo -e "${JIT}${GREEN} >> Installation stated${BLACK}"
	echo -e "${JIT}${GREEN} >>> Creating directory${BLACK}"
	mkdir -p ${INSTALL_LOCATION}

	# Clone GKFS
	echo -e "${JIT}${GREEN} >>> Installing GEKKO${BLACK}"
	cd ${INSTALL_LOCATION}
	git clone --recurse-submodules https://storage.bsc.es/gitlab/hpc/gekkofs.git
	cd gekkofs
	# git checkout main fmt10
	git pull --recurse-submodules
	cd ..

	# Workaround for lib fabric
	# echo -e "${RED}>>> Work around for libfabric${BLACK}"
	# sed  -i '/\[\"libfabric/d' ${INSTALL_LOCATION}/gekkofs/scripts/profiles/latest/default_zmq.specs
	# sed  -i 's/\"libfabric\"//g' ${INSTALL_LOCATION}/gekkofs/scripts/profiles/latest/default_zmq.specs

	# Build GKFS
	gekkofs/scripts/gkfs_dep.sh -p default_zmq ${INSTALL_LOCATION}/iodeps/git ${INSTALL_LOCATION}/iodeps
	cd gekkofs && mkdir build && cd build
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${INSTALL_LOCATION}/iodeps -DGKFS_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_LOCATION}/iodeps -DGKFS_ENABLE_CLIENT_METRICS=ON ..
	make -j 4 install || echo -e "${RED}>>> Error encountered${BLACK}"
	echo -e "${JIT}${GREEN} >>> GEKKO installed${BLACK}"

	#Cardo DEPS: CEREAL
	echo -e "${JIT}${GREEN} >>> Installing Cargo${BLACK}"
	cd ${INSTALL_LOCATION}
	git clone https://github.com/USCiLab/cereal
	cd cereal && mkdir build && cd build
	cmake -DCMAKE_PREFIX_PATH=${INSTALL_LOCATION}/iodeps -DCMAKE_INSTALL_PREFIX=${INSTALL_LOCATION}/iodeps ..
	make -j 4 install || echo -e "${RED}>>> Error encountered${BLACK}"

	#Cargo DEPS: THALLIUM
	cd ${INSTALL_LOCATION}
	git clone https://github.com/mochi-hpc/mochi-thallium
	cd mochi-thallium && mkdir build && cd build
	cmake -DCMAKE_PREFIX_PATH=${INSTALL_LOCATION}/iodeps -DCMAKE_INSTALL_PREFIX=${INSTALL_LOCATION}/iodeps ..
	make -j 4 install || echo -e "${RED}>>> Error encountered${BLACK}"

	# clone cargo:EXCLUDE_FTIO=false
EXCLUDE_CARGO=false
EXCLUDE_DEMON=false
EXCLUDE_PROXY=false
	cd ${INSTALL_LOCATION}
	git clone https://storage.bsc.es/gitlab/hpc/cargo.git
	cd cargo
	# git checkout rnou/fmt10
	git checkout marc/nek5000 #fixes slow Cargo and verbosity
	cd ..

	# build cargo
	cd cargo && mkdir build && cd build
	vim +332 ../src/master.cpp #adapt regex path
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${INSTALL_LOCATION}/iodeps -DCMAKE_INSTALL_PREFIX=${INSTALL_LOCATION}/iodeps ..
	# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${INSTALL_LOCATION}/iodeps -DCMAKE_INSTALL_PREFIX=${INSTALL_LOCATION}/iodeps 	-DGKFS_BUILD_TESTS=ON  -DCARGO_TRANSPORT_LIBRARY:STRING=libfabric ..

	make -j 4 install || echo -e "${RED}>>> Error encountered${BLACK}"
	# GekkoFS should be found in the cargo CMAKE configuration.
	echo -e "${JIT}${GREEN} >>> Cargon installed${BLACK}"

	## build FTIO:
	# echo -e "${GREEN}>>> Installing FTIO${BLACK}"
	# cd ${INSTALL_LOCATION}
	# git clone https://github.com/tuda-hpclab/ftio.git
	# ml lang/Python/3.10.8-GCCcore-12.2.0 || echo "skipping module load";
	# cd FTIO
	# # Install FTIO
	# make install  || echo -e "${RED}>>> Error encountered${BLACK}"
	# echo -e "${GREEN}>>> FTIO installed${BLACK}"

	#build IOR
	cd ${INSTALL_LOCATION}
	git clone https://github.com/hpc/ior.git
	cd ior
	./bootstrap
	./configure
	make -j 4 #&& make -j 4 install

	echo -e "${JIT}${GREEN} >> Installation finished${BLACK}"
	echo -e "\n
	>> read to go <<
	call: ./jit.sh  -n NODES -t MAX_TIME 
    "

}

function parse_options() {
	# Define the options
	OPTIONS="a:r:n:c:p:o:t:j:l:i:e:xdyvh"
    LONGOPTS="address:,port:,nodes:,total_procs:,procs_list:,omp_threads:,max-time:,job-id:,log-name:,install-location:,exclude:,exclude-all,skip_confirm,dry_run,verbose,help"


	# Parse the options using getopt
    PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
    if [[ $? -ne 0 ]]; then
        error_usage
        exit 2
    fi
    eval set -- "$PARSED"
    # Process options
    while true; do
        case "$1" in
        -a | --address)
            ADDRESS_FTIO="$2"
            shift 2
            ;;
        -r | --port)
            PORT="$2"
            shift 2
            ;;
        -n | --nodes)
            NODES="$2"
            shift 2
            ;;
        -c | --total_procs)
            PROCS="$2"
            shift 2
            ;;
        -p | --procs_list)
            PROCS_LIST="$2"
            shift 2
            ;;
        -o | --omp_threads)
            OMP_THREADS="$2"
            shift 2
            ;;
        -t | --max-time)
            MAX_TIME="$2"
            shift 2
            ;;
        -j | --job-id)
            JOB_ID="$2"
            STATIC_ALLOCATION=true
            shift 2
            ;;
        -l | --log-name)
            LOG_DIR="$2"
            shift 2
            ;;
        -i | --install-location)
            INSTALL_LOCATION="$2"
            install_all
            shift 2
            ;;
        -e | --exclude)
            echo -e "${GREEN}>>${YELLOW} Excluding: "
            if [[ -z "$2" || "$2" == -* ]]; then
                EXCLUDE_FTIO=true
                echo -e "- ftio"
                [[ "$2" == -* ]] && shift || shift 2
            else
                IFS=',' read -ra EXCLUDES <<< "$2"
                for exclude in "${EXCLUDES[@]}"; do
                    case "$exclude" in
                    ftio)
                        EXCLUDE_FTIO=true
                        echo -e "- ftio"
                        ;;
                    cargo)
                        EXCLUDE_CARGO=true
                        echo -e "- cargo"
                        ;;
                    gkfs)
                        EXCLUDE_DEMON=true
                        EXCLUDE_PROXY=true
                        echo -e "- gkfs"
                        ;;
                    demon)
                        EXCLUDE_DEMON=true
                        echo -e "- demon"
                        ;;
                    proxy)
                        EXCLUDE_PROXY=true
                        echo -e "- proxy"
                        ;;
                    all)
                        EXCLUDE_ALL=true
                        echo -e "- all"
                        ;;
                    *)
                        echo -e "${RED}Invalid exclude option: $exclude ${BLACK}" >&2
                        exit 1
                        ;;
                    esac
                done
                shift 2
            fi
            echo -e "${BLACK}"
            ;;
        -x | --exclude-all)
            EXCLUDE_ALL=true
            shift
            ;;
        -d | --dry_run)
            DRY_RUN=true
            shift
            ;;
        -v | --verbose)
            VERBOSE=true
            shift
            ;;
        -y | --skip_confirm)
            SKIP_CONFIRM=true
            shift
            ;;
        -h | --help)
            error_usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo -e "${RED}Invalid option: -$1 ${BLACK}" >&2
            error_usage
            exit 1
            ;;
        esac
    done

    # Additional logic to handle procs_list
    if [[ -n "$PROCS_LIST" ]]; then
        IFS=',' read -ra PROCS_ARRAY <<< "$PROCS_LIST"
        if [[ ${#PROCS_ARRAY[@]} -gt 5 ]]; then
            echo -e "${RED}Too many values for --procs_list. Maximum is 5.${BLACK}" >&2
            exit 1
        fi

        # Assign values to specific variables
        for i in "${!PROCS_ARRAY[@]}"; do
            case $i in
                0) PROCS_APP="${PROCS_ARRAY[$i]}" ;;
                1) PROCS_DEMON="${PROCS_ARRAY[$i]}" ;;
                2) PROCS_PROXY="${PROCS_ARRAY[$i]}" ;;
                3) PROCS_CARGO="${PROCS_ARRAY[$i]}" ;;
                4) PROCS_FTIO="${PROCS_ARRAY[$i]}" ;;
            esac
        done
    fi
}


function find_time() {

	n=$1
	tm=0
	if [ $n -lt 101 ]; then
		tm=05
	elif [ $n -lt 1001 ]; then
		tm=30
	elif [ $n -lt 5001 ]; then
		tm=10
	elif [ $n -lt 10001 ]; then
		tm=30
	else
		#tm=$(($n / 100))
		tm=60
	fi

	return $tm
}

function shut_down() {
	local name=$1
	local PID=$2
	echo "Shutting down ${name} with PID ${PID}"
	if [[ -n ${PID} ]]; then
		# kill -s SIGINT ${PID} &
		kill ${PID}

		# wait works only for local
		if [ "$CLUSTER" == true ]; then
			wait ${PID}
		fi
	fi
}

function get_id() {
	trueIdid=$(squeue | grep "APPXGEKKO" | awk '{print $(1) }')
	return $trueIdid
}

function info() {
	trueID=$(get_id)
	echo -e "${BLUE}\nWorking directory is -> $PWD ${BLACK}"
	echo -e "${BLUE}Target nodes ---------> $NODES ${BLACK}"
	echo -e "${BLUE}Max time -------------> $MAX_TIME ${BLACK}"
	echo -e "${BLUE}Job id ---------------> ${RED}$trueId \n ${BLACK}"
}

wait_for_gkfs_daemons() {
	sleep 2
	local server_wait_cnt=0
	local nodes=${NODES}
	until [ $(($(wc -l "${GKFS_HOSTFILE}" 2>/dev/null | awk '{print $1}') + 0)) -eq "${nodes}" ]; do
		sleep 2
		server_wait_cnt=$((server_wait_cnt + 1))
		if [ ${server_wait_cnt} -gt 600 ]; then
			echo "Server failed to start. Exiting ..."
			exit 1
		fi
	done
}

function create_hostfile() {
	echo -e "${JIT}${CYAN} >> Cleaning Hostfile: ${GKFS_HOSTFILE} ${BLACK}"
	rm -f ${GKFS_HOSTFILE} || echo -e "${BLUE}>> No hostfile found ${BLACK}"

	# echo -e "${CYAN}>> Creating Hostfile: ${GKFS_HOSTFILE} ${BLACK}"
	# touch ${GKFS_HOSTFILE}
	# for i in "${NODES_ARR[@]::${#NODES_ARR[@]}-1}"; do #exclude last element as this is FTIO_NODE
	# 	echo "cpu$i" >> ${GKFS_HOSTFILE}
	# done
}

function check_error_free() {
	if [ $? -eq 0 ]; then
		echo -e "${JIT}${GREEN} >> $1 successful ${BLACK}"
	else
		echo -e "${JIT}${RED} >> $1 failed! Exiting ${BLACK}<<<" >&2
		exit 1
	fi
}

function progress() {
	Animationflag=0
	trueID=$(get_id)
	status=$(squeue | grep $trueId | awk '{print $5 }' | tail -1)
	job_nodes=$(squeue | grep $trueId | awk '{print $(9) }' | tail -1)
	time_limit=$(squeue | grep $trueId | awk '{print $(7) }' | tail -1)
	while [[ $status != "C" ]] && [[ $status != "F" ]] && [[ $status != "S" ]] && [[ ! -z $status ]]; do
		if [[ $status == *R* ]]; then
			if [[ flag -eq 0 ]]; then
				start_time="$(date -u +%s)"
				flag=1
				echo -e "\n${BLUE}  RUNNING on $job_nodes nodes${BLACK}"
			fi
			end_time="$(date -u +%s)"
			echo -en "\r${CYAN}  Running --> elapsed time:[ $(($end_time - $start_time)) / $(($((10#$tm)) * 60 + $((10#$th)) * 3600)) ] seconds ${BLACK}"

		elif [[ $status == *PD* ]]; then
			if [[ $Animationflag -eq 0 ]]; then
				echo -en "\r  PENDING  \  "
				Animationflag=1
			elif [[ $Animationflag -eq 1 ]]; then
				echo -en "\r  PENDING  |  "
				Animationflag=2
			elif [[ $Animationflag -eq 2 ]]; then
				echo -en "\r  PENDING  /  "
				Animationflag=3
			else
				echo -en "\r  PENDING  -  "
				Animationflag=0
			fi

		elif [[ $status == *C* ]]; then
			echo -e "\n  CONFIGURING  "

		elif [[ $status == *F* ]]; then
			echo "  --FAILED--"
			#break
			exit [0]

		else
			echo "  $status"
		fi

		# Sleep for few seconds
		#if [[ $((tm / 5)) -eq 0 ]]; then
		if [[ $MAX_TIME -lt 60 ]]; then
			sleep 1
		else
			sleep $(echo "($MAX_TIME)/20" | bc -l)
		fi

		status=$(squeue | grep $trueId | awk '{print $5 }' | tail -1)
	done

	echo -e "${GREEN}\n  ---- Simulation complete ----${BLACK}"

	time=$(squeue | grep $trueId | awk '{print $(6) }' | tail -1)
	if [[ -z "$time" ]]; then
		time=$(squeue | awk '{print $(6) }' | tail -1)
	fi
	echo -e "${BLUE}  Finished run with $job_nodes nodes in $time / $time_limit${BLACK}"
}

function get_address_ftio() {
	# get Address and port
	echo -e "\n${BLUE}####### Getting FTIO ADDRESS ${BLACK}"
	if [ "$CLUSTER" == true ]; then
		local call="srun --jobid=${JOB_ID} ${FTIO_NODE_COMMAND} --disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 ip addr | grep ib0 | awk '{print \$2}'| cut -d'/' -f1 | tail -1"
		echo -e "${JIT}${CYAN} >> Executing: ${call} ${BLACK}"
		ADDRESS_FTIO=$(eval " ${call}")
		# ADDRESS_FTIO=$(eval " ${call}" | awk '{print $NF}')
	else
		echo "using default address"
	fi

	echo -e "${JIT} ${GREEN}>> ADDRESS_FTIO: ${ADDRESS_FTIO}${BLACK}"
}

function get_address_cargo() {
	echo -e "\n${BLUE}####### Getting Cargo ADDRESS ${BLACK}"
	if [ "$CLUSTER" == true ]; then
		local call="srun --jobid=${JOB_ID}  ${SINGLE_NODE_COMMAND} --disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 ip addr | grep ib0 | awk '{print \$2}'| cut -d'/' -f1 | tail -1"
		echo -e "${JIT}${CYAN} >> Executing: ${call} ${BLACK}"
		ADDRESS_CARGO=$(eval " ${call}")
		CARGO_SERVER="ofi+sockets://${ADDRESS_CARGO}:62000"
	else
		CARGO_SERVER="ofi+tcp://${ADDRESS_CARGO}:62000"
	fi

	echo -e "${JIT} ${GREEN}>> ADDRESS_CARGO: ${ADDRESS_CARGO}${BLACK}"
	echo -e "${JIT} ${GREEN}>> CARGO_SERVER:  ${CARGO_SERVER}${BLACK}"
}

function format_time() {
    local input_time="$1"
    local int_time="${input_time%.*}"
    local nanos="${input_time#*.}"

    # Calculate hours, minutes, and seconds using shell arithmetic
    local h=$((int_time / 3600))
    local m=$(( (int_time % 3600) / 60 ))
    local s=$((int_time % 60))

    # Format nanos to ensure it's exactly 9 digits long
    nanos=$(printf "%-9s" "$nanos" | tr ' ' '0')

    # Format and output the time as HH:MM:SS.NNNNNNNNN
    local out=$(printf "%02d:%02d:%02d.%s" "$h" "$m" "$s" "$nanos")
    echo "${out}"
}


# Function to print settings
function print_settings() {

    local ftio_status="${GREEN}ON${BLACK}"
    local gkfs_demon_status="${GREEN}ON${BLACK}"
    local gkfs_proxy_status="${GREEN}ON${BLACK}"
    local cargo_status="${GREEN}ON${BLACK}"

    local task_demon="${APP_NODES}"
    local cpu_demon="${PROCS_DEMON}"
    local task_proxy="${APP_NODES}"
    local cpu_proxy="${PROCS_PROXY}"
    local task_cargo="${APP_NODES} * ${PROCS_CARGO}"
    local cpu_cargo="${PROCS_CARGO}"
    local task_ftio="1"
    local cpu_ftio="${PROCS_FTIO}"

    local ftio_text="
├─ ftio location  : ${BLACK}${FTIO_BIN_LOCATION}${BLACK}
├─ address ftio   : ${BLACK}${ADDRESS_FTIO}${BLACK}
├─ port           : ${BLACK}${PORT}${BLACK}
├─ # nodes        : ${BLACK}1${BLACK}
└─ ftio node      : ${BLACK}${FTIO_NODE_COMMAND##'--nodelist='}${BLACK}"

    local gkfs_demon_text="
├─ gkfs demon     : ${BLACK}${GKFS_DEMON}${BLACK}
├─ gkfs intercept : ${BLACK}${GKFS_INTERCEPT}${BLACK}
├─ gkfs mntdir    : ${BLACK}${GKFS_MNTDIR}${BLACK}
├─ gkfs rootdir   : ${BLACK}${GKFS_ROOTDIR}${BLACK}
├─ gkfs hostfile  : ${BLACK}${GKFS_HOSTFILE}${BLACK}"

    local gkfs_proxy_text="
├─ gkfs proxy     : ${BLACK}${GKFS_PROXY}${BLACK}
└─ gkfs proxyfile : ${BLACK}${GKFS_PROXYFILE}${BLACK}"

    local cargo_text="
├─ cargo          : ${BLACK}${CARGO}${BLACK}
├─ cargo cli      : ${BLACK}${CARGO_CLI}${BLACK}
├─ stage in path  : ${BLACK}${STAGE_IN_PATH}${BLACK}
└─ address cargo  : ${BLACK}${ADDRESS_CARGO}${BLACK}"

    if [ "$EXCLUDE_FTIO" == true ] ; then
        ftio_text="
├─ ftio location  : ${YELLOW}none${BLACK}
├─ address ftio   : ${YELLOW}none${BLACK}
├─ port           : ${YELLOW}none${BLACK}
├─ # nodes        : ${YELLOW}none${BLACK}
└─ ftio node      : ${YELLOW}none${BLACK}"
        ftio_status="${YELLOW}OFF${BLACK}"
        task_ftio="${YELLOW}-[/]"
        cpu_ftio="${YELLOW}-[/]"
    fi

    if [ "${EXCLUDE_DEMON}" == true ]; then
        gkfs_demon_text="
├─ gkfs demon     : ${YELLOW}none${BLACK}
├─ gkfs intercept : ${YELLOW}none${BLACK}
├─ gkfs mntdir    : ${YELLOW}none${BLACK}
├─ gkfs rootdir   : ${YELLOW}none${BLACK}
├─ gkfs hostfile  : ${YELLOW}none${BLACK}"
        gkfs_demon_status="${YELLOW}OFF${BLACK}"
        task_demon="${YELLOW}-[/]"
        cpu_demon="${YELLOW}-[/]"
    fi

    if [ "${EXCLUDE_PROXY}" == true ]; then
        gkfs_proxy_text="
├─ gkfs proxy     : ${YELLOW}none${BLACK}
└─ gkfs proxyfile : ${YELLOW}none${BLACK}"
        gkfs_proxy_status="${YELLOW}OFF${BLACK}"
        task_proxy="${YELLOW}-[/]"
        cpu_proxy="${YELLOW}-[/]"
    fi

    if [ "${EXCLUDE_CARGO}" == true ]; then
        cargo_text="
├─ cargo location : ${YELLOW}none${BLACK}
├─ cargo cli      : ${YELLOW}none${BLACK}
├─ stage in path  : ${YELLOW}none${BLACK}
└─ address cargo  : ${YELLOW}none${BLACK}"
        cargo_status="${YELLOW}OFF${BLACK}"
        task_cargo="${YELLOW}-[/]"
        cpu_cargo="${YELLOW}-[/]"
    fi

    local text="
${JIT} ${GREEN}Settings      
##################${BLACK}
${GREEN}setup${BLACK}
├─ logs dir       : ${BLACK}${LOG_DIR}${BLACK}
├─ pwd            : ${BLACK}$(pwd)${BLACK}
├─ ftio           : ${ftio_status}
├─ gkfs demon     : ${gkfs_demon_status}
├─ gkfs proxy     : ${gkfs_proxy_status}
├─ cargo          : ${cargo_status}
├─ cluster        : ${BLACK}${CLUSTER}${BLACK}
├─ total nodes    : ${BLACK}${NODES}${BLACK}
│   ├─ app        : ${BLACK}${APP_NODES}${BLACK}
│   └─ ftio       : ${BLACK}1${BLACK}
├─ tasks per node : -  
│   ├─ app        : ${BLACK}${PROCS_APP}${BLACK} 
│   ├─ demon      : ${BLACK}${task_demon}${BLACK}
│   ├─ proxy      : ${BLACK}${task_proxy}${BLACK}
│   ├─ cargo      : ${BLACK}${task_cargo}${BLACK}
│   └─ ftio       : ${BLACK}${task_ftio}${BLACK}
├─ cpus per task  : ${BLACK}${PROCS}${BLACK} 
│   ├─ app        : ${BLACK}1${BLACK}
│   ├─ demon      : ${BLACK}${cpu_demon}${BLACK}
│   ├─ proxy      : ${BLACK}${cpu_proxy}${BLACK}
│   ├─ cargo      : ${BLACK}${cpu_cargo}${BLACK}
│   └─ ftio       : ${BLACK}${cpu_ftio}${BLACK}
├─ OMP threads    : ${BLACK}${OMP_THREADS}${BLACK}
├─ max time       : ${BLACK}${MAX_TIME}${BLACK}
└─ job id         : ${BLACK}${JOB_ID}${BLACK}

${GREEN}ftio${BLACK}${ftio_text}

${GREEN}gekko${BLACK}${gkfs_demon_text}${gkfs_proxy_text}

${GREEN} cargo${BLACK}${cargo_text}

${GREEN}app${BLACK}
├─ app dir        : ${BLACK}${APP_DIR}${BLACK}
├─ app call       : ${BLACK}${APP_CALL}${BLACK}
├─ # nodes        : ${BLACK}${APP_NODES}${BLACK}
└─ app nodes      : ${BLACK}${APP_NODES_COMMAND##'--nodelist='}${BLACK}
${GREEN}##################${BLACK}
"

    # Print to console
    echo -e "${text}"

    # Save to settings.log
    echo -e "${text}" > "${LOG_DIR}/settings.log"

    # Verify if settings.log was created and has content
    # echo "Log file content:"
    # cat "${LOG_DIR}/settings.log"
}

# Function to calculate and display elapsed time
function elapsed_time() {
    local name="$1"
    local start_time="$2"
    local end_time="$3"

    # Calculate runtime in seconds
	local runtime=$(awk "BEGIN {print $end_time - $start_time}")
    
    # Format runtime
    local runtime_formatted=$(format_time "${runtime}")

    # Print elapsed time information
    echo -e "\n\n${BLUE}############${JIT}${BLUE}##############\n# ${name}\n# time:${BLACK} ${runtime_formatted} ${BLUE}\n#${BLACK} ${runtime} ${BLUE} seconds\n##############################${BLACK}\n\n" | tee -a "${LOG_DIR}/time.log"
}

function log_dir() {
	# Define default LOG_DIR if not set
    if [[ -z "${LOG_DIR}" ]]; then
        LOG_DIR="logs_nodes${NODES}_Jobid${JOB_ID}"
        if [[ -n "${LOG_SUFFIX}" ]]; then
            LOG_DIR+="_${LOG_SUFFIX}"
        fi
    fi

    # Create directory if it does not exist
    mkdir -p "${LOG_DIR}"

    # Resolve and return the absolute path of LOG_DIR
    LOG_DIR=$(realpath "${LOG_DIR}")

    # Define log file paths
    GEKKO_DEMON_LOG="${LOG_DIR}/gekko_demon.log"
    GEKKO_DEMON_ERR="${LOG_DIR}/gekko_demon.err"
    GEKKO_PROXY_LOG="${LOG_DIR}/gekko_proxy.log"
    GEKKO_PROXY_ERR="${LOG_DIR}/gekko_proxy.err"
    GEKKO_CLIENT_LOG="${LOG_DIR}/gekko_client.log"
    CARGO_LOG="${LOG_DIR}/cargo.log"
    CARGO_ERR="${LOG_DIR}/cargo.err"
    FTIO_LOG="${LOG_DIR}/ftio.log"
    FTIO_ERR="${LOG_DIR}/ftio.err"
    APP_LOG="${LOG_DIR}/app.log"
    APP_ERR="${LOG_DIR}/app.err"
}

function execute() {
	local exec_call=$1

	if [[ $# -eq 2 ]]; then
		echo -e "${JIT}${CYAN} >> $2 ${BLACK}"
	elif [[ $# -eq 3 ]]; then
		echo -e "${JIT}$2$3${BLACK}"
	else
		echo -e "${JIT}${CYAN} >> Executing: ${exec_call} ${BLACK}"
	fi

	eval "${exec_call}"
}

function execute_and_wait_msg() {
	local exec_call="$1"
	# message to stop the command at
	local message="$4"
	#save the size of the cargo log file
	local last_pos=$(stat -c %s "${LOG_DIR}/cargo.log")
	echo -e "${JIT}"$2""$3"${BLACK}"
	eval " ${call}"
	# id=$(eval " ${call}")
	# id=$(echo $id  | grep -oP '(?<=server replied with: )\d+')

	while true; do
		current_pos=$(stat -c %s "${LOG_DIR}/cargo.log")
		if ((current_pos > last_pos)); then
			LAST_LINE=$(tail -n 1 "${LOG_DIR}/cargo.log")
			if [[ "$LAST_LINE" == *"$message"* ]]; then
				echo -e "${JIT}${CYAN} >> End of transfer detected ${BLACK}"
				break
			fi
		fi
	done
}

function execute_and_wait_stage_out() {
	local exec_call="$1"
	# message to stop the command at
	message="$4"
	#save the size of the cargo log file

	# Store the current position in the log file
	local start_lines=$(wc -l <"${LOG_DIR}/cargo.log")

	echo -e "${JIT}"$2""$3"${BLACK}"
	eval " ${call}"

	echo -e "${JIT}${CYAN} >> Waiting for  '${message}' ${BLACK}"

	while true; do
		current_lines=$(wc -l <"${LOG_DIR}/cargo.log")
		if ((current_lines > start_lines)); then
			n_lines=$((${current_lines} - ${start_lines}))
			files=$(tail -n ${n_lines} "${LOG_DIR}/cargo.log")
			if [[ "${files}" == *"${message}"* ]]; then
				echo -e "${JIT}${CYAN} >> End of transfer detected ${BLACK}"
				break
			fi
		fi
	done

	files=$(echo "${files}" | grep "${message}")
	if [[ "${files}" == *"Transfer finished for []"* ]]; then
		return
	else
		local last_file=$(echo "${files}" | grep -oP '(?<=path: ").*?(?=")' | tail -1)
		echo -e "${JIT}${CYAN} >> Waiting for deletion of ${last_file} ${BLACK}"
		while true; do
			LAST_LINE=$(tail -n 1 "${LOG_DIR}/cargo.log")
			if [[ "$LAST_LINE" == *"Deleting ${last_file}"* ]]; then
				echo -e "${JIT}${CYAN} >> End of call detected ${BLACK}"
				break
			fi
		done
	fi
}

function wait_msg() {
	local log="$1"
	local message="$2"
	echo -e "${JIT}${CYAN} >> Waiting for detection of ${message} ${BLACK}"
	local last_pos=$(stat -c %s "${log}")
	while true; do
		current_pos=$(stat -c %s "${log}")
		if ((current_pos > last_pos)); then
			LAST_LINE=$(tail -n 1 "${log}")
			if [[ "$LAST_LINE" == *"$message"* ]]; then
				echo -e "${JIT}${CYAN} >> End of transfer detected ${BLACK}"
				break
			fi
		fi
	done
}

function cancel_jit_jobs() {
	if [ -n "$(hostname | grep 'cpu\|mogon')" ]; then
		# Get the list of job IDs with the name "JIT"
		jit_jobs=$(squeue --me --name=JIT --format=%A | tail -n +2)

		if [ -z "$jit_jobs" ]; then
			return
		fi

		echo -e "${JIT}${YELLOW} >> The following jobs with the name 'JIT' were found:\n $jit_jobs"

		# Prompt the user to confirm cancellation
		read -p "Do you want to cancel all 'JIT' jobs? (yes/no): " confirmation

		if [[ "$confirmation" == "yes" || "$confirmation" == "y" || "$confirmation" == "ye" ]]; then
			for job_id in $jit_jobs; do
				scancel "$job_id"
				echo -e "${JIT}${CYAN} >> Cancelled job ID $job_id"
			done
			echo -e "${JIT}${GREEN} >> All 'JIT' jobs have been cancelled."
		else
			echo -e "${JIT}${YELLOW} >> No jobs were cancelled."
		fi
	fi
}

function get_pid() {
	local name=$1
	local pid=$2
	if [ "$CLUSTER" == true ]; then
		pid=$(ps aux | grep "srun" | grep "${JOB_ID}" | grep "$1" | grep -v grep | tail -1 | awk '{print $2}')
	fi

	if [[ "${name}" == "${CARGO}" ]]; then
		CARGO_PID=${pid}
		echo -e "${JIT}${GREEN} CARGO PID: ${CARGO_PID} ${BLACK}"
	elif [[ "${name}" == "${GKFS_DEMON}" ]]; then
		GEKKO_DEMON_PID=${pid}
		echo -e "${JIT}${GREEN} GEKKO_DEMON PID: ${GEKKO_DEMON_PID} ${BLACK}"
	elif [[ "${name}" == "${GKFS_PROXY}" ]]; then
		GEKKO_PROXY_PID=${pid}
		echo -e "${JIT}${GREEN} GEKKO_PROXY PID: ${GEKKO_PROXY_PID} ${BLACK}"
	elif  [[ "${name}" == "FTIO" ]] || [[ "${name}" == "predictor_jit" ]] ; then
		FTIO_PID=${pid}
		echo -e "${JIT}${GREEN} FTIO PID: ${FTIO_PID} ${BLACK}"
	fi
}

function relevant_files() {
	# if [ "$CLUSTER" == true ]; then
		if [[ $# -eq 1 ]]; then
			echo -e "${JIT}${CYAN} >> Setting up ignored files${BLACK}"
		fi

		local call="echo -e '${REGEX_MATCH}' > ${REGEX_FILE}"
		if [[ $# -eq 1 ]]; then
			execute "${call}"
		else
			eval "${call}"
		fi

		if [[ $# -eq 1 ]]; then
			echo -e "${JIT}${CYAN} >> Files that match ${REGEX_MATCH} are ignored ${BLACK}"
		fi
		echo -e "${JIT}${CYAN} >> cat ${REGEX_FILE}: \n$(cat ${REGEX_FILE}) ${BLACK}\n"
	# fi

}

function reset_relevant_files() {

	if [ "$CLUSTER" == true ]; then
		echo -e "${JIT}${CYAN} >> Reseting ignored files${BLACK}"
		local call="echo -e  '.*' > ${REGEX_FILE}"
		execute "${call}"
		# echo -e "${JIT}${CYAN} >> cat ${REGEX_FILE}: \n$(cat ${REGEX_FILE}) ${BLACK}\n"
	fi

}

function total_time() {
	local tot_time=$(cat ${LOG_DIR}/time.log | grep "seconds" | awk {'print $2'} | awk '{s+=$1} END {print s}')
	echo -e "\n${JIT}${GREEN} >>\n       App time: ${APP_TIME}\n  Stage in time: ${STAGE_IN_TIME}\n Stage out time: ${STAGE_OUT_TIME}\n---------------- ${BLACK}"
	echo -e "${JIT} --> Total Time: ${tot_time}${BLACK}\n" | tee -a ${LOG_DIR}/time.log
}


function check_setup(){
	# check settings
	if [ "${EXCLUDE_ALL}" == false ]; then
		echo -e "${JIT}${CYAN} >> MPI hostfile:${BLACK}\n$(cat ${APP_DIR}/hostfile_mpi) ${BLACK}\n"
		echo -e "${JIT}${CYAN} >> Gekko hostfile:${BLACK}\n$(cat ${GKFS_HOSTFILE}) ${BLACK}\n"
		local additional_arguments=$(generate_additional_arguments)
		
		local call="${additional_arguments} ls ${GKFS_MNTDIR}"
		local files=$(${call})
		echo -e "${JIT}${CYAN} >> ${call}:${BLACK}\n${files} ${BLACK}\n"
		
		# echo -e "${JIT}${CYAN} >> statx:${BLACK}\n"
		# mpiexec -np ${PROCS} --oversubscribe --hostfile ${APP_DIR}/hostfile_mpi --map-by node -x LIBGKFS_LOG=errors -x LD_PRELOAD=${GKFS_INTERCEPT} -x LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} -x LIBGKFS_PROXY_PID_FILE=${GKFS_PROXYFILE} /home/tarrafah/nhr-admire/tarraf/FTIO/ftio/api/gekkoFs/scripts/test.sh
		
		# echo -e "${JIT}${CYAN} >> SESSION.NAME:${BLACK}\n$(cat /lustre/project/nhr-admire/tarraf/admire/turbPipe/run_gkfs/SESSION.NAME) ${BLACK}\n"
	
		# Tersting
		# local files2=$(srun --export=LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE},LD_LIBRARY_PATH=${LD_LIBRARY_PATH},LD_PRELOAD=${GKFS_INTERCEPT} --jobid=${JOB_ID} ${APP_NODES_COMMAND} --disable-status -N ${APP_NODES}  --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0  /usr/bin/ls ${GKFS_MNTDIR} )
		# echo -e "${JIT}${CYAN} >> srun ls ${GKFS_MNTDIR}:${BLACK}\n${files2} ${BLACK}\n"
		# local files3=$(mpiexec -np 1 --oversubscribe --hostfile ${APP_DIR}/hostfile_mpi --map-by node -x LIBGKFS_LOG=errors -x LIBGKFS_ENABLE_METRICS=on -x LIBGKFS_METRICS_IP_PORT=${ADDRESS_FTIO}:${PORT} -x LD_PRELOAD=${GKFS_INTERCEPT} -x LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} -x LIBGKFS_PROXY_PID_FILE=${GKFS_PROXYFILE} /usr/bin/ls ${GKFS_MNTDIR} )
		# echo -e "${JIT}${CYAN} >> mpirun ls ${GKFS_MNTDIR}:${BLACK}\n${files3} ${BLACK}\n"
	fi

	sleep 1
}

function set_flags(){
	if [ "${EXCLUDE_ALL}" == true ]; then
		EXCLUDE_FTIO=true
		EXCLUDE_CARGO=true
		EXCLUDE_DEMON=true
		EXCLUDE_PROXY=true
		APP_TIME="0"
		STAGE_IN_TIME="0"
		STAGE_OUT_TIME="0"
	fi
}

generate_additional_arguments() {
    additional_arguments=""

    # Check if FTIO is not excluded
    if [ "$EXCLUDE_FTIO" != true ]; then
        additional_arguments+="LIBGKFS_METRICS_IP_PORT=${ADDRESS_FTIO}:${PORT} LIBGKFS_ENABLE_METRICS=on "
    fi

    # Check if Proxy is not excluded
    if [ "$EXCLUDE_PROXY" != true ]; then
        additional_arguments+="LIBGKFS_PROXY_PID_FILE=${GKFS_PROXYFILE} "
    fi

    # Check if Demon is not excluded
    if [ "$EXCLUDE_DEMON" != true ]; then
        additional_arguments+="LIBGKFS_LOG=info,warnings,errors "
        additional_arguments+="LIBGKFS_LOG_OUTPUT=${GEKKO_CLIENT_LOG} "
        additional_arguments+="LIBGKFS_HOSTS_FILE=${GKFS_HOSTFILE} "
        additional_arguments+="LD_PRELOAD=${GKFS_INTERCEPT} "
    fi

    echo "$additional_arguments"
}