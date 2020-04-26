Network torchvision.models.googlenet {
Layer Conv2d-1 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 64, C: 3, R: 7, S: 7, Y: 224, X: 224 }
}
Layer Conv2d-2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 64, R: 3, S: 3, Y: 56, X: 56 }
}
Layer Conv2d-4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 192, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-5 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 96, C: 192, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-6 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 96, R: 3, S: 3, Y: 28, X: 28 }
}
Layer Conv2d-7 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 16, C: 192, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-8 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 32, C: 16, R: 3, S: 3, Y: 28, X: 28 }
}
Layer Conv2d-9 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 32, C: 192, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-10 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 256, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-11 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 256, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-12 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 128, R: 3, S: 3, Y: 28, X: 28 }
}
Layer Conv2d-13 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 32, C: 256, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-14 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 96, C: 32, R: 3, S: 3, Y: 28, X: 28 }
}
Layer Conv2d-15 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 256, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-16 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 480, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-17 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 96, C: 480, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-18 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 208, C: 96, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-19 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 16, C: 480, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-20 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 48, C: 16, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-21 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 480, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-22 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 512, R: 1, S: 1, Y: 4, X: 4 }
}
Layer Linear-23 {
Type: CONV
Dimensions { K: 1024, C: 2048, R: 1, S: 1, Y: 1, X: 1 }
}
Layer Linear-24 {
Type: CONV
Dimensions { K: 1000, C: 1024, R: 1, S: 1, Y: 1, X: 1 }
}
Layer Conv2d-25 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 160, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-26 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 112, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-27 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 224, C: 112, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-28 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 24, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-29 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 24, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-30 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-31 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-32 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-33 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 128, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-34 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 24, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-35 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 24, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-36 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-37 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 112, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-38 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 144, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-39 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 288, C: 144, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-40 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 32, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-41 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 32, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-42 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 512, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-43 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 528, R: 1, S: 1, Y: 4, X: 4 }
}
Layer Linear-44 {
Type: CONV
Dimensions { K: 1024, C: 2048, R: 1, S: 1, Y: 1, X: 1 }
}
Layer Linear-45 {
Type: CONV
Dimensions { K: 1000, C: 1024, R: 1, S: 1, Y: 1, X: 1 }
}
Layer Conv2d-46 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 528, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-47 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 160, C: 528, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-48 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 320, C: 160, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-49 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 32, C: 528, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-50 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 32, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-51 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 528, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-52 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 832, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Conv2d-53 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 160, C: 832, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Conv2d-54 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 320, C: 160, R: 3, S: 3, Y: 7, X: 7 }
}
Layer Conv2d-55 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 32, C: 832, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Conv2d-56 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 32, R: 3, S: 3, Y: 7, X: 7 }
}
Layer Conv2d-57 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 832, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Conv2d-58 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 384, C: 832, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Conv2d-59 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 832, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Conv2d-60 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 384, C: 192, R: 3, S: 3, Y: 7, X: 7 }
}
Layer Conv2d-61 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 48, C: 832, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Conv2d-62 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 48, R: 3, S: 3, Y: 7, X: 7 }
}
Layer Conv2d-63 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 832, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Linear-64 {
Type: CONV
Dimensions { K: 1000, C: 1024, R: 1, S: 1, Y: 1, X: 1 }
}
}