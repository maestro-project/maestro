./maestro --num_pes=256 \
          --l1_size_cstr=100 \
          --l2_size_cstr=3000 \
          --offchip_bw_cstr=50 \
          --Mapping_file='data/mapping/Resnet50_kcp_ws.m' \
          --print_res=true \
          --print_res_csv_file=true \
          --print_log_file=false \
