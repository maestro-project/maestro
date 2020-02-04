Network vgg16 {
Layer CONV1 { 
Type: CONV
Dimensions { K 64,C 3,R 3,S 3,Y 224,X 224 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
}
}

Layer CONV2 { 
Type: CONV
Dimensions { K 64,C 64,R 3,S 3,Y 224,X 224 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
}
}

Layer CONV3 { 
Type: CONV
Dimensions { K 128,C 64,R 3,S 3,Y 112,X 112 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
}
}

Layer CONV4 { 
Type: CONV
Dimensions { K 128,C 128,R 3,S 3,Y 112,X 112 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
}

Layer CONV5 { 
Type: CONV
Dimensions { K 256,C 128,R 3,S 3,Y 56,X 56 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
}

Layer CONV6 { 
Type: CONV
Dimensions { K 256,C 256,R 3,S 3,Y 56,X 56 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
}

Layer CONV7 { 
Type: CONV
Dimensions { K 256,C 256,R 3,S 3,Y 56,X 56 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
}

Layer CONV8 { 
Type: CONV
Dimensions { K 512,C 256,R 3,S 3,Y 28,X 28 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
}

Layer CONV9 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 28,X 28 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
}

Layer CONV10 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 28,X 28 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
}

Layer CONV11 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
}

Layer CONV12 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
}

Layer CONV13 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(64,64) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
			TemporalMap(1,1) Y';
			TemporalMap(1,1) X';	
			Cluster(64, P);
			SpatialMap(1,1) K;
			TemporalMap(3,3) R;
			TemporalMap(3,3) S;
		}
}


}
