

Network ResNeXt50 {
	Layer CONV1 {
		Type: CONV
		Stride { X: 2, Y: 2 }		
		Dimensions { K: 64, C: 3, R: 1, S: 1, Y:224, X:224 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV2_1_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 128, C: 64, R: 1, S: 1, Y: 56, X: 56 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV2_1_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 4, C: 128, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV2_1_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 256, C: 128, R: 1, S: 1, Y: 56, X: 56 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV2_2_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 128, C: 256, R: 1, S: 1, Y: 56, X: 56 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV2_2_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 4, C: 128, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV2_2_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 256, C: 128, R: 1, S: 1, Y: 56, X: 56 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}


	Layer CONV2_3_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 128, C: 256, R: 1, S: 1, Y: 56, X: 56 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV2_3_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 4, C: 128, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV2_3_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 256, C: 128, R: 1, S: 1, Y: 56, X: 56 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}



	Layer CONV3_1_1 {
		Type: CONV
		Stride { X: 2, Y: 2 }
		Dimensions { K: 256, C: 256, R: 1, S: 1, Y: 56, X: 56 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV3_1_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 8, C: 256, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV3_1_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 512, C: 256, R: 1, S: 1, Y: 28, X: 28 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}


	Layer CONV3_2_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 256, C: 256, R: 1, S: 1, Y: 28, X: 28 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV3_2_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 8, C: 256, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV3_2_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 512, C: 256, R: 1, S: 1, Y: 28, X: 28 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}


	Layer CONV3_3_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 256, C: 256, R: 1, S: 1, Y: 28, X: 28 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV3_3_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 8, C: 256, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV3_3_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 512, C: 256, R: 1, S: 1, Y: 28, X: 28 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}


	Layer CONV3_4_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 256, C: 256, R: 1, S: 1, Y: 28, X: 28 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV3_4_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 8, C: 256, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV3_4_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 512, C: 256, R: 1, S: 1, Y: 28, X: 28 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}



	Layer CONV4_1_1 {
		Type: CONV
		Stride { X: 2, Y: 2 }
		Dimensions { K: 512, C: 512, R: 1, S: 1, Y: 28, X: 28 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV4_1_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 16, C: 512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV4_1_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 1024, C: 512, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}



	Layer CONV4_2_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 512, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV4_2_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 16, C: 512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV4_2_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 1024, C: 512, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}



	Layer CONV4_3_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 512, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV4_3_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 16, C: 512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV4_3_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 1024, C: 512, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}


	Layer CONV4_4_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 512, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV4_4_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 16, C: 512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV4_4_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 1024, C: 512, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}



	Layer CONV4_5_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 512, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV4_5_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 16, C: 512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV4_5_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 1024, C: 512, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}


	Layer CONV4_6_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 512, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV4_6_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 16, C: 512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV4_6_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 1024, C: 512, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}


	Layer CONV5_1_1 {
		Type: CONV
		Stride { X: 2, Y: 2 }
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV5_1_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 32, C: 1024, R: 3, S: 3, Y: 7, X: 7 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV5_1_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 2048, C: 1024, R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}



	Layer CONV5_2_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 1024, C: 2048, R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV5_2_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 32, C: 1024, R: 3, S: 3, Y: 7, X: 7 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV5_2_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 2048, C: 1024, R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}


	Layer CONV5_3_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 1024, C: 2048, R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	Layer CONV5_3_2 {
		Type: NGCONV
		Stride { X: 1, Y: 1 }
		Dimensions { G: 32, K: 32, C: 1024, R: 3, S: 3, Y: 7, X: 7 }
		Dataflow {
			TemporalMap(1,1) G;
			SpatialMap(1,1) K;
			TemporalMap(28,28) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(28, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV5_3_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 2048, C: 1024, R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(256,256) C;
			Cluster(256, P);
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;			
            SpatialMap(1,1) C;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}


	Layer FC1000 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 1000, C: 2048, R: 7, S: 7, Y: 7, X: 7 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(5,5) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;			
			Cluster(5, P);
            SpatialMap(1,1) C;
			Cluster(Sz(R), P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S), P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}




}

