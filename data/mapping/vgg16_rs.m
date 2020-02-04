Network VGG16 {
	Layer CONV1 {
		Type: CONV
		Dimensions { K: 64, C:3, R: 3, S: 3, Y: 224, X: 224 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV2 {
		Type: CONV
		Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 224, X: 224 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV3 {
		Type: CONV
		Dimensions { K: 128, C: 64, R: 3, S: 3, Y: 112, X: 112 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV4 {
		Type: CONV
		Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 112, X: 112 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}


	Layer CONV5 {
		Type: CONV
		Dimensions { K: 256, C:128, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV6 {
		Type: CONV
		Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV7 {
		Type: CONV
		Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV8 {
		Type: CONV
		Dimensions { K: 512, C: 256, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV9 {
		Type: CONV
		Dimensions { K: 512, C:512, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV10 {
		Type: CONV
		Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV11 {
		Type: CONV
		Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV12 {
		Type: CONV
		Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}

	Layer CONV13 {
		Type: CONV
		Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			TemporalMap(2,2) K;
			TemporalMap(3,3) C;
			TemporalMap(3,3) R;			
			SpatialMap(3,1) Y;
			TemporalMap(3,1) X;
			Cluster(3,P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) R;
			TemporalMap(3,3) S;
		}
	}
}


//Accelerator {
//  PE { NumPEs: 128; VectorWidth: 4; MultPrecision: INT8, AddPrecision: INT16 }
//  Buffer { GlobalL2: 2048, LocalL1: 64}
//  NoC {Bandwidth: 64; AvgLatency: 2}
//}
