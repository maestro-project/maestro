// BLAS 3 - Dense Matrix-Dense Matrix multiplication
// Constants are in GEMM convention; (MxK matrix) x (KxN matrix) = (MxN matrix)
Constant SzB 10;
Constant SzM 100;
Constant SzN 100;
Constant SzK 100;

// Optimize for data reuse
// Constant MTileSz 10;
// Constant NTileSz 50;
// Constant KTileSz 100;

// Optimize for throughput
Constant BTileSz 2;
Constant MTileSz 1;
Constant NTileSz 1;
Constant KTileSz 100;


Network TestBMM {
	Layer BMM {
		Type: BMM
		Dimensions { B: SzB, K: SzK, M: SzM, N: SzN }
		Dataflow {
			TemporalMap(BTileSz, BTileSz) B;
			TemporalMap(NTileSz,NTileSz) N; 
			SpatialMap(MTileSz,MTileSz) M;
			TemporalMap(KTileSz,KTileSz) K;
			Cluster(KTileSz, P);
			TemporalMap(BTileSz, BTileSz) B;			
			TemporalMap(NTileSz,NTileSz) N;			
			TemporalMap(MTileSz,MTileSz) M;
			SpatialMap(1,1) K;
		}
	}
}
