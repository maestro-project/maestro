Network VGG16 {

	Layer CONV1 {
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { K: 64, C:3, R: 3, S: 3, Y: 224, X: 224 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV2 {
		Type: CONV
		Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 224, X: 224 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}

	}

	Layer CONV3 {
		Type: CONV
		Dimensions { K: 128, C: 64, R: 3, S: 3, Y: 112, X: 112 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}


	Layer CONV4 {
		Type: CONV
		Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 112, X: 112 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV5 {
		Type: CONV
		Dimensions { K: 256, C:128, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}



	Layer CONV6 {
		Type: CONV
		Dimensions { K: 256, C: 256, Y: 56, X: 56, R: 3, S: 3 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV7 {
		Type: CONV
		Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV8 {
		Type: CONV
		Dimensions { K: 512, C: 256, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV9 {
		Type: CONV
		Dimensions { K: 512, C:512, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV10 {
		Type: CONV
		Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 28, X: 28 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV11 {
		Type: CONV
		Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}

	Layer CONV12 {
		Type: CONV
		Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 14, X: 14 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}


	Layer CONV13 {
		Type: CONV
		Dimensions { K: 512, C: 512, Y: 14, X: 14, R: 3, S: 3 }
		Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X';     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(Sz(R),Sz(R)) R; // This cannot be changed
			TemporalMap(Sz(S),Sz(S)) S; // This cannot be changed
			// Virtual Neuron of size Sz(R) x Sz(S):
			Cluster(Sz(R),P);
			SpatialMap(1,1) Y;
			SpatialMap(1,1) R;
			Cluster(Sz(S),P);
			SpatialMap(1,1) X;
			SpatialMap(1,1) S;
		}
	}
}
