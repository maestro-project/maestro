Network Resnet34 {
	Layer CONV1 {
		Type: CONV
		Dimensions { K: 64, C:3, R: 7, S: 7, Y: 224, X: 224 }
		Dataflow {
			TemporalMap(7,2) Y;
			TemporalMap(7,2) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(7,7) R;
			TemporalMap(7,7) S;
		}
	}

	Layer CONV2_1_s1 {
		Type: CONV
		Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 112, X: 112 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV2_2_s1 {
		Type: CONV
		Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV2_1_s2 {
		Type: CONV
		Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV2_2_s2 {
		Type: CONV
		Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV2_1_s3 {
		Type: CONV
		Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV2_2_s3 {
		Type: CONV
		Dimensions { K: 128, C: 64, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV3_1_s1 {
		Type: CONV
		Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV3_2_s1 {
		Type: CONV
		Dimensions { K: 128, C:128, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV3_1_s2 {
		Type: CONV
		Dimensions { K: 128, C:128, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV3_2_s2 {
		Type: CONV
		Dimensions { K: 128, C:128, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV3_1_s3 {
		Type: CONV
		Dimensions { K: 128, C:128, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV3_2_s3 {
		Type: CONV
		Dimensions { K: 128, C:128, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV3_1_s4 {
		Type: CONV
		Dimensions { K: 128, C:128, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV3_2_s4 {
		Type: CONV
		Dimensions { K: 256, C:128, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV4_1_s1 {
		Type: CONV
		Dimensions { K: 256, C:256, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV4_2_s1 {
		Type: CONV
		Dimensions { K: 256, C:256, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV4_1_s2 {
		Type: CONV
		Dimensions { K: 256, C:256, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV4_2_s2 {
		Type: CONV
		Dimensions { K: 256, C:256, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV4_1_s3 {
		Type: CONV
		Dimensions { K: 256, C:256, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV4_2_s3 {
		Type: CONV
		Dimensions { K: 256, C:256, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV4_1_s4 {
		Type: CONV
		Dimensions { K: 256, C:256, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV4_2_s4 {
		Type: CONV
		Dimensions { K: 256, C:256, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV4_1_s5 {
		Type: CONV
		Dimensions { K: 256, C:256, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV4_2_s5 {
		Type: CONV
		Dimensions { K: 256, C:256, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV4_1_s6 {
		Type: CONV
		Dimensions { K: 256, C:256, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV4_2_s6 {
		Type: CONV
		Dimensions { K: 512, C:256, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV5_1_s1 {
		Type: CONV
		Dimensions { K: 512, C:512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV5_2_s1 {
		Type: CONV
		Dimensions { K: 512, C:512, R: 3, S: 3, Y: 7, X: 7 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV5_1_s2 {
		Type: CONV
		Dimensions { K: 512, C:512, R: 3, S: 3, Y: 7, X: 7 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV5_2_s2 {
		Type: CONV
		Dimensions { K: 512, C:512, R: 3, S: 3, Y: 7, X: 7 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV5_1_s3 {
		Type: CONV
		Dimensions { K: 512, C:512, R: 3, S: 3, Y: 7, X: 7 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV5_2_s3 {
		Type: CONV
		Dimensions { K: 512, C:512, R: 3, S: 3, Y: 7, X: 7 }
		Dataflow {
			TemporalMap(3,1) Y;
			TemporalMap(3,1) X;
			TemporalMap(1,1) C;
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
	}
}


//Accelerator {
//  PE { NumPEs: 128; VectorWidth: 4; MultPrecision: INT8, AddPrecision: INT16 }
//  Buffer { GlobalL2: 2048, LocalL1: 64}
//  NoC {Bandwidth: 64; AvgLatency: 2}
//}
