#!/usr/bin/env bash

set -e

# Check if container path is valid
if [ ! -f $1 ]; then
	echo ERROR: invalid container path.
	exit 1
fi

# Parse hpc.yaml configuration file
script_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
source $script_dir/parse_yaml.sh
eval $(parse_yaml apptainer/hpc.yaml)

# Define additional shell variables
repository_name=${PWD##*/}
container_path=$1
container_name=${container_path##*/}
container_directory=${container_name%.*}
commit=$(echo $container_directory | cut -d "_" -f 4)

# Check parsed configuration
if [ -z "$job_name" ]; then
	echo ERROR: job_name not defined in hpc.yaml.
	exit 1
else
	pbs_job_name="#PBS -N $job_name"
fi
if [ -z "$walltime" ]; then
	echo ERROR: walltime not defined in hpc.yaml.
	exit 1
else
	pbs_walltime="#PBS -l walltime=$walltime"
fi
if [ -z "$resources" ]; then
	echo ERROR: resources not defined in hpc.yaml.
	exit 1
else
	pbs_resources="#PBS -l $resources"
fi
if [ -z "$job_array" ]; then
	echo ERROR: job_array not defined in hpc.yaml.
	exit 1
elif [ $job_array == "null" ]; then
	pbs_job_array=""
else
	pbs_job_array="#PBS -J $job_array"
fi
if [ -z "$kwargs" ]; then
	echo ERROR: kwargs not defined in hpc.yaml.
	exit 1
elif [ "$kwargs" == "null" ]; then
	kwargs=""
fi
if [ $app == "null" ]; then
	app=""
else
	app="--app $app"
fi

# Create temporary directory
tmp_dir=$(mktemp -d)

# Define HPC directory
hpc_dir=\~/projects/$repository_name/output/$container_directory/

# Send container to the HPC
rsync --verbose --ignore-existing --progress --rsync-path="mkdir -p $hpc_dir && rsync" -e ssh $container_path $hpc:$hpc_dir

# Create jobscripts
table="|Job ID|Job Name|Job Script|Status|args\n"
counter=1
for args in $args_; do
	# Expand args
	args=$(eval echo \$${args})

	# Build jobscript from template
	tmp_jobscript=$(mktemp -p $tmp_dir)
	sed "s/{{ pbs_job_name }}/$pbs_job_name/g" $script_dir/template.job > $tmp_jobscript
	sed -i "s/{{ pbs_walltime }}/$pbs_walltime/g" $tmp_jobscript
	sed -i "s/{{ pbs_resources }}/$pbs_resources/g" $tmp_jobscript
	sed -i "s/{{ pbs_job_array }}/$pbs_job_array/g" $tmp_jobscript
	sed -i "s/{{ repository_name }}/$repository_name/g" $tmp_jobscript
	sed -i "s/{{ container_directory }}/$container_directory/g" $tmp_jobscript
	sed -i "s/{{ container_name }}/$container_name/g" $tmp_jobscript
	sed -i "s/{{ app }}/$app/g" $tmp_jobscript
	sed -i "s/{{ commit }}/$commit/g" $tmp_jobscript
	sed -i "s/{{ args }}/$args/g" $tmp_jobscript

	# Send jobscript to the HPC
	rsync --quiet -e ssh $tmp_jobscript $hpc:$hpc_dir
	jobid=$(ssh $hpc "cd $hpc_dir && /opt/pbs/bin/qsub $kwargs $hpc_dir/${tmp_jobscript##*/} 2> /dev/null")

	# Rename jobscript to $jobid.job
	if [ $? == 0 ]; then
		ssh $hpc "mv $hpc_dir/${tmp_jobscript##*/} $hpc_dir/${job_name}_${jobid%.*}.job"
		table+="$counter|${jobid%.*}|$job_name|${job_name}_${jobid%.*}.job|Queued|$args\n"
	else
		ssh $hpc "rm $hpc_dir/${tmp_jobscript##*/}"
		table+="$counter|-|-|-|Failed|$args\n"
	fi

	counter=$((counter + 1))
done

rm -rf $tmp_dir
echo
echo -e $table | column -s '|' -t
