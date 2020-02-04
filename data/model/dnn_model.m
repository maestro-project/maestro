Network sequential {
Layer conv2d {
Type: CONV
Stride { X: 4, Y: 4 }
Dimensions { K: 96, C: 3, R: 11, S: 11, Y: 224, X: 224 }
}
Layer conv2d_1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 96, R: 5, S: 5, Y: 26, X: 26 }
}
Layer conv2d_2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 384, C: 256, R: 3, S: 3, Y: 10, X: 10 }
}
Layer conv2d_3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 384, C: 384, R: 3, S: 3, Y: 8, X: 8 }
}
Layer conv2d_4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 384, R: 3, S: 3, Y: 6, X: 6 }
}
Layer dense {
Type: CONV
Dimensions { K: 4096, C: 256, R: 1, S: 1, Y: 1, X: 1 }
}
Layer dense_1 {
Type: CONV
Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1, X: 1 }
}
Layer dense_2 {
Type: CONV
Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1, X: 1 }
}
Layer dense_3 {
Type: CONV
Dimensions { K: 1, C: 4096, R: 1, S: 1, Y: 1, X: 1 }
}
}