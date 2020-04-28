//Make some assumptions
// let's assume seq_len = 64 for now
// N d_mod d_ff h dk dv 
// 6 512   2048 8 64 64
Constant Seq_Len 128;

Network Transformer {
	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_1 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	//~~~~~~~~~~~~~~~End encoder
	//Start decoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_1 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	// ~~~~ End Decoder MH1 layer ~~~~

	//Just plit into VK, Q
	Layer MH_FC_DimReduce_VK_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1024, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_Q_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_1 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 512, one for each VKQ
		Dimensions { N: Seq_Len, K: 1536, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_01 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_02 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	Layer SD_MatMul_QK_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_03 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_04 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_QK_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 64, S: 1, Y:64, X:Seq_Len }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	Layer SD_MatMul_V_05 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:64 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 512->2048
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 2048, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 2048, Y:1, X:2048 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good
	

	/// done with h parallel sd layers now
	Layer FinalLinear { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 512, C: 1, R: 1, S: 512, Y:1, X:512 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}


}

