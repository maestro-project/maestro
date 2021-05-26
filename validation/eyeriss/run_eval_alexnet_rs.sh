./maestro \
          --Mapping_file='alexnet_rs_validation.m' \
          --print_res=true \
          --print_res_csv_file=true \
          --print_log_file=false \
             --noc_bw=1000000000 \
          --noc_hop_latency=1 \
          --noc_mc_support=true \
          --num_pes=168 \
          --num_simd_lanes=1 \
          --l1_size=5120 \
          --l2_size=1024000 \
          --do_implicit_reduction=false \
          --do_fg_sync=false \
          --do_reduction_op=true \
          --do_dse=false \


