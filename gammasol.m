
Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 64, C: 64, Y: 56, X: 56, R: 3, S: 3 }
Dataflow {
TemporalMap(55,55) X';
TemporalMap(1,1) K;
SpatialMap(32,32) C;
TemporalMap(56,56) Y';
TemporalMap(3,3) R;
TemporalMap(3,3) S;
Cluster(56,P);
TemporalMap(3,3) C;
TemporalMap(1,1) R;
TemporalMap(1,1) S;
TemporalMap(5,5) X';
SpatialMap(1,1) Y';
TemporalMap(1,1) K;
}
}
		}
