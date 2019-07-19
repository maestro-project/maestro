
Network MobileNetV2 {
	Layer CONV1 {
		Type: CONV
		Stride { X: 2, Y: 2 }		
		Dimensions { K: 32, C: 3, R: 1, S: 1, Y:224, X:224 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

//Bottleneck 1
//BottleneckID_repeat_operatorID

	Layer Bottleneck1_1_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 32, C: 32, R: 1, S: 1, Y:112, X:112 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck1_1_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 32, R: 3, S: 3, Y:112, X:112 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck1_1_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 16, C: 32, R: 1, S: 1, Y:112, X:112 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck1_1_Residual {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 16, R: 1, S: 1, Y:112, X:112 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


// Bottleneck 2
	Layer Bottleneck2_1_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 96, C: 16, R: 1, S: 1, Y:112, X:112 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck2_1_2 {
		Type: DSCONV
		Stride { X: 2, Y: 2 }		
		Dimensions { K: 1, C: 96, R: 3, S: 3, Y:112, X:112 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck2_1_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 96, C: 96, R: 1, S: 1, Y:56, X:56 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck2_2_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 96, C: 96, R: 1, S: 1, Y:56, X:56 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck2_2_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 96, R: 3, S: 3, Y:56, X:56 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck2_2_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 24, C: 96, R: 1, S: 1, Y:56, X:56 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


//Bottleneck 3


	Layer Bottleneck3_1_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 144, C: 24, R: 1, S: 1, Y:56, X:56 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck3_1_2 {
		Type: DSCONV
		Stride { X: 2, Y: 2 }		
		Dimensions { K: 1, C: 144, R: 3, S: 3, Y:56, X:56 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck3_1_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 144, C: 144, R: 1, S: 1, Y:28, X:28 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}



	Layer Bottleneck3_2_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 144, C: 144, R: 1, S: 1, Y:28, X:28 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck3_2_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 144, R: 3, S: 3, Y:28, X:28 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck3_2_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 144, C: 144, R: 1, S: 1, Y:28, X:28 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


	Layer Bottleneck3_2_Residual {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 144, R: 1, S: 1, Y:28, X:28 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


	Layer Bottleneck3_3_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 144, C: 144, R: 1, S: 1, Y:28, X:28 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck3_3_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 144, R: 3, S: 3, Y:28, X:28 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck3_3_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 32, C: 144, R: 1, S: 1, Y: 28, X: 28 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


//Bottleneck 4


	Layer Bottleneck4_1_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 192, C: 32, R: 1, S: 1, Y: 28, X: 28 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck4_1_2 {
		Type: DSCONV
		Stride { X: 2, Y: 2 }		
		Dimensions { K: 1, C: 192, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck4_1_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 192, C: 192 R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}



	Layer Bottleneck4_2_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 192, C: 192, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck4_2_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 192, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck4_2_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 192, C: 192 R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck4_2_Residual {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 192, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


	Layer Bottleneck4_3_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 192, C: 192, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck4_3_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 192, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck4_3_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 192, C: 192 R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck4_3_Residual {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 192, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


	Layer Bottleneck4_4_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 192, C: 192, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck4_4_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 192, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck4_4_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 64, C: 192 R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


// Bottleneck 5

	Layer Bottleneck5_1_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 384, C: 64, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck5_1_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 384, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck5_1_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 384, C: 384 R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


	Layer Bottleneck5_2_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 384, C: 384, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck5_2_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 384, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck5_2_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 96, C: 384 R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

// Bottleneck 6

	Layer Bottleneck6_1_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 576, C: 96, R: 1, S: 1, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck6_1_2 {
		Type: DSCONV
		Stride { X: 2, Y: 2 }		
		Dimensions { K: 1, C: 576, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck5_2_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 576, C: 576 R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}



	Layer Bottleneck6_2_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 576, C: 576, R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck6_2_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 576, R: 3, S: 3, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck6_2_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 576, C: 576 R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


	Layer Bottleneck6_2_Residual {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 576 R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


	Layer Bottleneck6_3_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 576, C: 576, R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck6_3_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 576, R: 3, S: 3, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck6_3_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 160, C: 576 R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

// Bottleneck 7

	Layer Bottleneck7_1_1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 960, C: 160, R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck7_1_2 {
		Type: DSCONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 1, C: 960, R: 3, S: 3, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}

	Layer Bottleneck7_1_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 320, C: 960, R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


// CONV2D

	Layer CONV2D_2 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 1280, C: 320, R: 1, S: 1, Y: 7, X: 7 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}


	Layer CONV2D_3 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 1000, C: 1280, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (Sz(R), 1) Y;
			TemporalMap (Sz(S), 1) X;
			TemporalMap (Sz(R), Sz(R)) R;
			TemporalMap (Sz(S), Sz(S)) S;
			SpatialMap (1,1) C;
		}
	}



}
