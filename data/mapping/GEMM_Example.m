// BLAS 3 - Dense Matrix-Dense Matrix multiplication
// Constants are in GEMM convention; (MxK matrix) x (KxN matrix) = (MxN matrix)
Constant SzM 100;
Constant SzN 100;
Constant SzK 100;

// Optimize for data reuse
// Constant MTileSz 10;
// Constant NTileSz 50;
// Constant KTileSz 100;

// Optimize for throughput
Constant MTileSz 1;
Constant NTileSz 1;
Constant KTileSz 100;


Network BLAS3 {
	Layer BLAS {
		Type: GEMM
		Dimensions { K: SzK, M: SzM, N: SzN }
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

// Network BLAS3 {
//	Layer BLAS {
//		Type: CONV
//		Stride { X: 1, Y: 1 }
//		Dimensions { K: SzN, C: 1, R: 1, S: SzK, Y: SzM, X: SzK }
//		Dataflow {
//			TemporalMap(NTileSz,NTileSz) K;
//			SpatialMap(MTileSz,MTileSz) Y;
//			TemporalMap(SzK,SzK) X;
//			TemporalMap(SzK,SzK) S;
//			Cluster(1,P);
//			SpatialMap(KTileSz,KTileSz) X;
//			TemporalMap(KTileSz,KTileSz) S;			
//			Cluster(KTileSz,P);
//			SpatialMap(1,1) X;
//			SpatialMap(1,1) S;
//		}
// 	}
// }