Network torchvision.models.mnasnet {
Layer Conv2d-1 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 32, C: 3, R: 3, S: 3, Y: 224, X: 224 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-2 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 32, R: 3, S: 3, Y: 112, X: 112 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 16, C: 32, R: 1, S: 1, Y: 112, X: 112 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 48, C: 16, R: 1, S: 1, Y: 112, X: 112 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-5 {
Type: DSCONV
Stride { X: 2, Y: 2 }
Dimensions { K: 1, C: 48, R: 3, S: 3, Y: 112, X: 112 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-6 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 24, C: 48, R: 1, S: 1, Y: 56, X: 56 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-7 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 72, C: 24, R: 1, S: 1, Y: 56, X: 56 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-8 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 72, R: 3, S: 3, Y: 56, X: 56 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-9 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 24, C: 72, R: 1, S: 1, Y: 56, X: 56 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-10 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 72, C: 24, R: 1, S: 1, Y: 56, X: 56 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-11 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 72, R: 3, S: 3, Y: 56, X: 56 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-12 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 24, C: 72, R: 1, S: 1, Y: 56, X: 56 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-13 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 72, C: 24, R: 1, S: 1, Y: 56, X: 56 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-14 {
Type: DSCONV
Stride { X: 2, Y: 2 }
Dimensions { K: 1, C: 72, R: 5, S: 5, Y: 56, X: 56 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-15 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 40, C: 72, R: 1, S: 1, Y: 28, X: 28 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-16 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 120, C: 40, R: 1, S: 1, Y: 28, X: 28 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-17 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 120, R: 5, S: 5, Y: 28, X: 28 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-18 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 40, C: 120, R: 1, S: 1, Y: 28, X: 28 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-19 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 120, C: 40, R: 1, S: 1, Y: 28, X: 28 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-20 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 120, R: 5, S: 5, Y: 28, X: 28 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-21 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 40, C: 120, R: 1, S: 1, Y: 28, X: 28 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-22 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 240, C: 40, R: 1, S: 1, Y: 28, X: 28 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-23 {
Type: DSCONV
Stride { X: 2, Y: 2 }
Dimensions { K: 1, C: 240, R: 5, S: 5, Y: 28, X: 28 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-24 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 80, C: 240, R: 1, S: 1, Y: 14, X: 14 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-25 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 480, C: 80, R: 1, S: 1, Y: 14, X: 14 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-26 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 480, R: 5, S: 5, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-27 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 80, C: 480, R: 1, S: 1, Y: 14, X: 14 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-28 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 480, C: 80, R: 1, S: 1, Y: 14, X: 14 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-29 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 480, R: 5, S: 5, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-30 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 80, C: 480, R: 1, S: 1, Y: 14, X: 14 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-31 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 480, C: 80, R: 1, S: 1, Y: 14, X: 14 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-32 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 480, R: 3, S: 3, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-33 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 96, C: 480, R: 1, S: 1, Y: 14, X: 14 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-34 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 576, C: 96, R: 1, S: 1, Y: 14, X: 14 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-35 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 576, R: 3, S: 3, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-36 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 96, C: 576, R: 1, S: 1, Y: 14, X: 14 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-37 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 576, C: 96, R: 1, S: 1, Y: 14, X: 14 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-38 {
Type: DSCONV
Stride { X: 2, Y: 2 }
Dimensions { K: 1, C: 576, R: 5, S: 5, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-39 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 576, R: 1, S: 1, Y: 7, X: 7 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-40 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1152, C: 192, R: 1, S: 1, Y: 7, X: 7 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-41 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 1152, R: 5, S: 5, Y: 7, X: 7 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-42 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 1152, R: 1, S: 1, Y: 7, X: 7 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-43 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1152, C: 192, R: 1, S: 1, Y: 7, X: 7 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-44 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 1152, R: 5, S: 5, Y: 7, X: 7 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-45 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 1152, R: 1, S: 1, Y: 7, X: 7 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-46 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1152, C: 192, R: 1, S: 1, Y: 7, X: 7 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-47 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 1152, R: 5, S: 5, Y: 7, X: 7 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-48 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 1152, R: 1, S: 1, Y: 7, X: 7 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-49 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1152, C: 192, R: 1, S: 1, Y: 7, X: 7 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-50 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 1152, R: 3, S: 3, Y: 7, X: 7 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv2d-51 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 320, C: 1152, R: 1, S: 1, Y: 7, X: 7 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Conv2d-52 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1280, C: 320, R: 1, S: 1, Y: 7, X: 7 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
Layer Linear-53 {
Type: CONV
Dimensions { K: 1000, C: 1280, R: 1, S: 1, Y: 1, X: 1 }
Dataflow {
			TemporalMap (1,1) K;
			TemporalMap (1,1) C;
			TemporalMap (Sz(R),1) Y;
			SpatialMap (Sz(S),1) X;
			TemporalMap (Sz(R),Sz(R)) R;
			TemporalMap (Sz(S),Sz(S)) S;
}
}
}