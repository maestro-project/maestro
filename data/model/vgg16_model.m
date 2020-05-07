Network vgg16 {
Layer CONV1 { 
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K 64,C 3,R 3,S 3,Y 224,X 224 }
}

Layer CONV2 { 
Type: CONV
Dimensions { K 64,C 64,R 3,S 3,Y 224,X 224 }
}

Layer CONV3 { 
Type: CONV
Dimensions { K 128,C 64,R 3,S 3,Y 112,X 112 }

}

Layer CONV4 { 
Type: CONV
Dimensions { K 128,C 128,R 3,S 3,Y 112,X 112 }

}

Layer CONV5 { 
Type: CONV
Dimensions { K 256,C 128,R 3,S 3,Y 56,X 56 }

}

Layer CONV6 { 
Type: CONV
Dimensions { K 256,C 256,R 3,S 3,Y 56,X 56 }
}

Layer CONV7 { 
Type: CONV
Dimensions { K 256,C 256,R 3,S 3,Y 56,X 56 }

}

Layer CONV8 { 
Type: CONV
Dimensions { K 512,C 256,R 3,S 3,Y 28,X 28 }

}

Layer CONV9 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 28,X 28 }

}

Layer CONV10 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 28,X 28 }
}

Layer CONV11 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }
}


Layer CONV12 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }

}

Layer CONV13 { 
Type: CONV
Dimensions { K 512,C 512,R 3,S 3,Y 14,X 14 }

}


}
