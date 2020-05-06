Network gnmt_gemm {
Layer GEMM0 {
Type: CONV
Dimensions { K: 2048, C: 4096, Y: 128, X: 1, R: 1, S: 1 }
Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(14,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;
		}
}
Layer GEMM1 {
Type: CONV
Dimensions { K: 2048, C: 4096, Y: 128, X: 1, R: 1, S: 1 }
Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(14,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;
		}
}
Layer GEMM2 {
Type: CONV
Dimensions { K: 3072, C: 4096, Y: 320, X: 1, R: 1, S: 1 }
Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(14,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;
		}
}
Layer GEMM3 {
Type: CONV
Dimensions { K: 2048, C: 4096, Y: 128, X: 1, R: 1, S: 1 }
Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(14,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;
		}
}
Layer GEMM4 {
Type: CONV
Dimensions { K: 2048, C: 4096, Y: 128, X: 1, R: 1, S: 1 }
Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(14,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;
		}
}
Layer GEMM5 {
Type: CONV
Dimensions { K: 3072, C: 4096, Y: 320, X: 1, R: 1, S: 1 }
Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(14,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;
		}
}
Layer GEMM6 {
Type: CONV
Dimensions { K: 3072, C: 4096, Y: 320, X: 1, R: 1, S: 1 }
Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(14,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;
		}
}
Layer GEMM7 {
Type: CONV
Dimensions { K: 3072, C: 4096, Y: 320, X: 1, R: 1, S: 1 }
Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(14,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;
		}
}
Layer GEMM8 {
Type: CONV
Dimensions { K: 3072, C: 4096, Y: 320, X: 1, R: 1, S: 1 }
Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(14,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;
		}
}
}