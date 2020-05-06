Network ncf_gemm {
Layer GEMM0 {
Type: CONV
Dimensions { K: 128, C: 2048, Y: 256, X: 1, R: 1, S: 1 }


}
Layer GEMM1 {
Type: CONV
Dimensions { K: 64, C: 2048, Y: 128, X: 1, R: 1, S: 1 }


}
Layer GEMM2 {
Type: CONV
Dimensions { K: 256, C: 2048, Y: 256, X: 1, R: 1, S: 1 }


}
Layer GEMM3 {
Type: CONV
Dimensions { K: 256, C: 256, Y: 2048, X: 1, R: 1, S: 1 }


}
Layer GEMM4 {
Type: CONV
Dimensions { K: 256, C: 256, Y: 2048, X: 1, R: 1, S: 1 }


}
Layer GEMM5 {
Type: CONV
Dimensions { K: 128, C: 256, Y: 2048, X: 1, R: 1, S: 1 }


}
Layer GEMM6 {
Type: CONV
Dimensions { K: 256, C: 128, Y: 2048, X: 1, R: 1, S: 1 }


}
Layer GEMM7 {
Type: CONV
Dimensions { K: 128, C: 64, Y: 2048, X: 1, R: 1, S: 1 }


}
Layer GEMM8 {
Type: CONV
Dimensions { K: 64, C: 128, Y: 2048, X: 1, R: 1, S: 1 }


}
Layer GEMM9 {
Type: CONV
Dimensions { K: 1, C: 2048, Y: 128, X: 1, R: 1, S: 1 }


}
Layer GEMM10 {
Type: CONV
Dimensions { K: 1, C: 128, Y: 2048, X: 1, R: 1, S: 1 }


}
Layer GEMM11 {
Type: CONV
Dimensions { K: 128, C: 1, Y: 2048, X: 1, R: 1, S: 1 }


}
}