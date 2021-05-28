
Network Resnet50_layer3 {

	Layer CONV2_1_2 {
		Type: CONV
		Stride { X: 1, Y: 1 }		
		Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 56, X: 56 }
		Dataflow {
			TemporalMap(2,2) Y;
			TemporalMap(64,64) C;
                        SpatialMap(8,8) X;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) R;			
			TemporalMap(1,1) S;			
			Cluster(32, P);
			TemporalMap(1,1) Y;			
			TemporalMap(1,1) C;			
			TemporalMap(1,1) S;			
			TemporalMap(1,1) R;			
			SpatialMap(2, 2) K;
			TemporalMap(8,8) X;			
		}

	}

}
