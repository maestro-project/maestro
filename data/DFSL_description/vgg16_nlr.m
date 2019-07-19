Network vgg16 {
	Layer CONV1 { 
	Type: CONV
	Dimensions { K 64,C 3,R 3,S 3,Y 224,X 224 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV2 { 
	Type: CONV
	Dimensions { K 64,C 64,R 3,S 3,Y 224,X 224 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV3 { 
	Type: CONV
	Dimensions { K 128,C 64,R 3,S 3,Y 112,X 112 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV4 { 
	Type: CONV
	Dimensions { K 128,C 128,R 3,S 3,Y 112,X 112 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV5 { 
	Type: CONV
	Dimensions { K 256,C 128,R 3,S 3,Y 56,X 56 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV6 { 
	Type: CONV
	Dimensions { K 256,C 256,R 3,S 3,Y 56,X 56 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV7 { 
	Type: CONV
	Dimensions { K 256,C 256,R 3,S 3,Y 56,X 56 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV8 { 
	Type: CONV
	Dimensions { K 512,C 256,R 3,S 3,Y 28,X 28 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV9 { 
	Type: CONV
	Dimensions { K 512,C 512,R 3,S 3,Y 28,X 28 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV10 { 
	Type: CONV
	Dimensions { K 512,C 512,R 3,S 3,Y 28,X 28 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV11 { 
	Type: CONV
	Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV12 { 
	Type: CONV
	Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }
	Dataflow {
		TemporalMap (1,1) K;
		TemporalMap (Sz(R), 1) Y;
		TemporalMap (Sz(S), 1) X;
		TemporalMap (Sz(R), Sz(R)) R;
		TemporalMap (Sz(S), Sz(S)) S;
		SpatialMap (1,1) C;
	}
}

Layer CONV13 { 
	Type: CONV
	Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }
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
