# Source directory containing the files to be copied
val_dir="EC40-NTREX-val-data-bin"

# Destination base directory containing the shard folders
train_dir="fairseq-data-bin-sharded"

# Loop through each shard folder and copy the files
for shard in shard0 shard1 shard2 shard3 shard4
do
    # Construct the destination directory path
    dest_dir="$train_dir/$shard"

    # Check if the destination directory exists
    if [ -d "$dest_dir" ]; then
        # Copy all files from the source directory to the destination directory
        cp -r "$val_dir/"* "$dest_dir"
    else
        echo "Destination directory $dest_dir does not exist."
    fi
done