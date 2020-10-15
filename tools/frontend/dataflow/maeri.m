Dataflow {
			// Only one spatial map can be applied except on R and S
			TemporalMap(1,1) C;      // Flexible (Mapping size/offset, spatial/temporal)
			SpatialMap(1,1) K;       // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) Y’;     // Flexible (Mapping size/offset, spatial/temporal)
			TemporalMap(1,1) X’;     // Flexible (Mapping size/offset, spatial/temporal)
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
		