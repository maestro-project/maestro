./maestro \
          --Mapping_file='gammasolx.m' \
          --print_res=true \
          --print_res_csv_file=true \
          --print_log_file=false \
             --noc_bw=1000000000 \
          --noc_hop_latency=0 \
          --noc_mc_support=true \
          --num_pes=256 \
          --num_simd_lanes=1 \
          --l1_size=5120 \
          --l2_size=1024000 \
          --do_implicit_reduction=true \
          --do_fg_sync=false \
          --do_reduction_op=true \
          --do_dse=false \


