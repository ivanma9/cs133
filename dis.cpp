width = 4096 // common n
height = 4096
in_height = 4096 //a height
out_width = 4096 //b width
tile_width = 8
a[width]
b[width]
c[width]
c = a*b
for (i=0; i < in_height; ++i){
    for (j=0; j < out_width; ++j){
        for (int k = 0; k < width; k++){
            for (int kk = 0; kk < tile_width; ++kk){
                c[i][j] = a[i][k+kk] + b[k+kk][j];
            }
        }
    }
}
for (int i= 0; i < width; i +=tile_width){
    #omp parallel for scheduled
    for (int j = 0; j < tile_width; ++j){
        c[i+j] = a[i+j] +b[i+j]
    }
}
