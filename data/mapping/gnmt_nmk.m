Constant MTileSz 1;
Constant NTileSz 1;
Constant KTileSz 100;
Network gnmt_gemm {
Layer GEMM0 {
Type: GEMM
Dimensions { N: 2048, K: 4096, M: 128 }
Dataflow {
			TemporalMap(NTileSz,NTileSz) N; 
			SpatialMap(MTileSz,MTileSz) M;
			TemporalMap(KTileSz,KTileSz) K;
			Cluster(KTileSz, P);
			TemporalMap(NTileSz,NTileSz) N;			
			TemporalMap(MTileSz,MTileSz) M;
			SpatialMap(1,1) K;
		}
}
Layer GEMM1 {
Type: GEMM
Dimensions { N: 2048, K: 4096, M: 128 }
Dataflow {
			TemporalMap(NTileSz,NTileSz) N; 
			SpatialMap(MTileSz,MTileSz) M;
			TemporalMap(KTileSz,KTileSz) K;
			Cluster(KTileSz, P);
			TemporalMap(NTileSz,NTileSz) N;			
			TemporalMap(MTileSz,MTileSz) M;
			SpatialMap(1,1) K;
		}
}
Layer GEMM2 {
Type: GEMM
Dimensions { N: 3072, K: 4096, M: 320 }
Dataflow {
			TemporalMap(NTileSz,NTileSz) N; 
			SpatialMap(MTileSz,MTileSz) M;
			TemporalMap(KTileSz,KTileSz) K;
			Cluster(KTileSz, P);
			TemporalMap(NTileSz,NTileSz) N;			
			TemporalMap(MTileSz,MTileSz) M;
			SpatialMap(1,1) K;
		}
}
Layer GEMM3 {
Type: GEMM
Dimensions { N: 2048, K: 4096, M: 128 }
Dataflow {
			TemporalMap(NTileSz,NTileSz) N; 
			SpatialMap(MTileSz,MTileSz) M;
			TemporalMap(KTileSz,KTileSz) K;
			Cluster(KTileSz, P);
			TemporalMap(NTileSz,NTileSz) N;			
			TemporalMap(MTileSz,MTileSz) M;
			SpatialMap(1,1) K;
		}
}
Layer GEMM4 {
Type: GEMM
Dimensions { N: 2048, K: 4096, M: 128 }
Dataflow {
			TemporalMap(NTileSz,NTileSz) N; 
			SpatialMap(MTileSz,MTileSz) M;
			TemporalMap(KTileSz,KTileSz) K;
			Cluster(KTileSz, P);
			TemporalMap(NTileSz,NTileSz) N;			
			TemporalMap(MTileSz,MTileSz) M;
			SpatialMap(1,1) K;
		}
}
Layer GEMM5 {
Type: GEMM
Dimensions { N: 3072, K: 4096, M: 320 }
Dataflow {
			TemporalMap(NTileSz,NTileSz) N; 
			SpatialMap(MTileSz,MTileSz) M;
			TemporalMap(KTileSz,KTileSz) K;
			Cluster(KTileSz, P);
			TemporalMap(NTileSz,NTileSz) N;			
			TemporalMap(MTileSz,MTileSz) M;
			SpatialMap(1,1) K;
		}
}
Layer GEMM6 {
Type: GEMM
Dimensions { N: 3072, K: 4096, M: 320 }
Dataflow {
			TemporalMap(NTileSz,NTileSz) N; 
			SpatialMap(MTileSz,MTileSz) M;
			TemporalMap(KTileSz,KTileSz) K;
			Cluster(KTileSz, P);
			TemporalMap(NTileSz,NTileSz) N;			
			TemporalMap(MTileSz,MTileSz) M;
			SpatialMap(1,1) K;
		}
}
Layer GEMM7 {
Type: GEMM
Dimensions { N: 3072, K: 4096, M: 320 }
Dataflow {
			TemporalMap(NTileSz,NTileSz) N; 
			SpatialMap(MTileSz,MTileSz) M;
			TemporalMap(KTileSz,KTileSz) K;
			Cluster(KTileSz, P);
			TemporalMap(NTileSz,NTileSz) N;			
			TemporalMap(MTileSz,MTileSz) M;
			SpatialMap(1,1) K;
		}
}
Layer GEMM8 {
Type: GEMM
Dimensions { N: 3072, K: 4096, M: 320 }
Dataflow {
			TemporalMap(NTileSz,NTileSz) N; 
			SpatialMap(MTileSz,MTileSz) M;
			TemporalMap(KTileSz,KTileSz) K;
			Cluster(KTileSz, P);
			TemporalMap(NTileSz,NTileSz) N;			
			TemporalMap(MTileSz,MTileSz) M;
			SpatialMap(1,1) K;
		}
}
}