Constant MTileSz 1;
Constant NTileSz 1;
Constant KTileSz 100;

Network ncf_gemm {
Layer GEMM0 {
Type: GEMM
Dimensions { N: 128, K: 2048, M: 256 }
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
Dimensions { N: 64, K: 2048, M: 128 }
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
Dimensions { N: 256, K: 2048, M: 256 }
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
Dimensions { N: 256, K: 256, M: 2048 }
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
Dimensions { N: 256, K: 256, M: 2048 }
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
Dimensions { N: 128, K: 256, M: 2048 }
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
Dimensions { N: 256, K: 128, M: 2048 }
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
Dimensions { N: 128, K: 64, M: 2048 }
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
Dimensions { N: 64, K: 128, M: 2048 }
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
Layer GEMM9 {
Type: GEMM
Dimensions { N: 1, K: 2048, M: 128 }
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
Layer GEMM10 {
Type: GEMM
Dimensions { N: 1, K: 128, M: 2048 }
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
Layer GEMM11 {
Type: GEMM
Dimensions { N: 128, K: 1, M: 2048 }
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