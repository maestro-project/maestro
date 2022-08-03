Network alexnet {

Layer CONV1 { 
  Type: CONV
  Stride { X: 4, Y: 4 }   
  Dimensions { N 4, K 96, C 3, R 11, S 11, Y 227, X 227 }
  Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) N;
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    TemporalMap(1,1) N;
    TemporalMap(1,1) C;    
    TemporalMap(2,2) K;    
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
  }
}

Layer CONV2 { 
Type: CONV
Dimensions { N 4, K 256, C 48, R 5, S 5, Y 31, X 31 }
  Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) N;
    TemporalMap(2,2) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    TemporalMap(1,1) C;    
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
  }
}

Layer CONV3 { 
Type: CONV
Dimensions { N 4, K 384, C 256, R 3, S 3, Y 15, X 15 }
  Dataflow {
    TemporalMap(4,4) N;
    SpatialMap(16,16) K;
    TemporalMap(4,4) C;
    TemporalMap(13,13) Y';
    TemporalMap(15,13) X;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;    
    Cluster(13, P);
    TemporalMap(4,4) N;
    TemporalMap(16,16) K;
    TemporalMap(4,4) C;
    SpatialMap(1,1) Y';
    TemporalMap(15,13) X;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    TemporalMap(4,4) N;
    TemporalMap(4,4) C;
    TemporalMap(16,16) K;
    TemporalMap(13,13) X';    
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S), Sz(S)) S;    
  }
}

Layer CONV4 { 
Type: CONV
Dimensions { N 4, K 384,C 192, R 3, S 3, Y 15, X 15 }
  Dataflow {
    TemporalMap(4,4) N;   
    TemporalMap(6,6) C;
    SpatialMap(16,16) K;
    TemporalMap(13,13) Y';
    TemporalMap(13,13) X';    
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(2, P);
    TemporalMap(4,4) N;
    SpatialMap(3,3) C;
    TemporalMap(16,16) K;
    TemporalMap(13,13) Y';
    TemporalMap(1,1) X';    
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;     
    Cluster(13, P);
    TemporalMap(4,4) N;
    TemporalMap(3,3) C;
    TemporalMap(16,16) K;
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';        
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;    
    Cluster(Sz(R),P);
    TemporalMap(1,1) N;    
    TemporalMap(1,1) C;
    TemporalMap(1,1) K;    
    TemporalMap(1,1) X';    
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S), Sz(S)) S;    
  }
}

Layer CONV5 { 
Type: CONV
Dimensions { N 4, K 256, C 192 , R 3, S 3, Y 15, X 15 }
  Dataflow {
    TemporalMap(4,4) N;   
    TemporalMap(6,6) C;
    SpatialMap(16,16) K;
    TemporalMap(13,13) Y';
    TemporalMap(13,13) X';    
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(2, P);
    TemporalMap(4,4) N;
    SpatialMap(3,3) C;
    TemporalMap(16,16) K;
    TemporalMap(13,13) Y';
    TemporalMap(8,8) X';
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(13, P);
    TemporalMap(1,1) N;
    TemporalMap(3,3) C;
    TemporalMap(16,16) K;
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';    
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;    
    Cluster(Sz(R),P);
    TemporalMap(1,1) N;
    TemporalMap(1,1) C;
    TemporalMap(2,2) K;
    TemporalMap(1,1) X';
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S), Sz(S)) S;    
  }
}

}
