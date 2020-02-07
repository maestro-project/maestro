Network vgg16 {
Layer CONV1 { 
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K 64,C 3,R 3,S 3,Y 224,X 224 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) S;			
		}
}

Layer CONV2 { 
Type: CONV
Dimensions { K 64,C 64,R 3,S 3,Y 224,X 224 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}
}

Layer CONV3 { 
Type: CONV
Dimensions { K 128,C 64,R 3,S 3,Y 112,X 112 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}

}

Layer CONV4 { 
Type: CONV
Dimensions { K 128,C 128,R 3,S 3,Y 112,X 112 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}

}

Layer CONV5 { 
Type: CONV
Dimensions { K 256,C 128,R 3,S 3,Y 56,X 56 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}

}

Layer CONV6 { 
Type: CONV
Dimensions { K 256,C 256,R 3,S 3,Y 56,X 56 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}
}

Layer CONV7 { 
Type: CONV
Dimensions { K 256,C 256,R 3,S 3,Y 56,X 56 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}

}

Layer CONV8 { 
Type: CONV
Dimensions { K 512,C 256,R 3,S 3,Y 28,X 28 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}

}

Layer CONV9 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 28,X 28 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}

}

Layer CONV10 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 28,X 28 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}
}

Layer CONV11 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}
}


Layer CONV12 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}

}

Layer CONV13 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }
		Dataflow {
			TemporalMap(1,1) K;
			TemporalMap(1,1) C;
			SpatialMap(Sz(R), 1) Y;
			TemporalMap(10,8) X;
			Cluster(8, P);
			SpatialMap(Sz(S), 1) X;
			TemporalMap(Sz(R), Sz(R)) R;
			TemporalMap(Sz(R), Sz(R)) S;			
		}

}


}
