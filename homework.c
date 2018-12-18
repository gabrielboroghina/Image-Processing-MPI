// Gabriel Boroghina, 333CB
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define IMG_TYPE_GRAYSCALE 5
#define IMG_TYPE_COLOR 6

typedef struct {
    unsigned char channel[3]; // channel[0] = red, channel[1] = green, channel[2] = blue
}__attribute__((packed)) color_pixel;

typedef unsigned char grayscale_pixel; // light of the grayscale pixel

typedef struct {
    int type, height, width;
    unsigned char maxval;
    void **pixel;
} image;

typedef struct {
    int begin, end;
} bounds;

typedef struct {
    float div;
    float kernel[3][3];
} filter_type;

const filter_type smoothFilter = {9, {{1, 1, 1},
                                      {1, 1, 1},
                                      {1, 1, 1}}};

const filter_type blurFilter = {16, {{1, 2, 1},
                                     {2, 4, 2},
                                     {1, 2, 1}}};

const filter_type sharpenFilter = {3, {{0, -2, 0},
                                       {-2, 11, -2},
                                       {0, -2, 0}}};

const filter_type meanRemovalFilter = {1, {{-1, -1, -1},
                                           {-1, 9, -1},
                                           {-1, -1, -1}}};

const filter_type embossFilter = {1, {{0, 1, 0},
                                      {0, 0, 0},
                                      {0, -1, 0}}};

void loadImage(const char *fileName, image *img) {
    FILE *imageFile = fopen(fileName, "r");

    // read image's type
    char imageType;
    fscanf(imageFile, "%*c%c", &imageType);
    img->type = imageType - '0';

    // read image's dimensions
    fscanf(imageFile, "%i%i%hhu%*c", &img->width, &img->height, &img->maxval);

    int pixelSize = img->type == IMG_TYPE_COLOR ? 3 : 1;
    img->pixel = (void **) malloc(img->height * sizeof(void *));

    // read the pixel matrix
    for (int i = 0; i < img->height; i++) {
        img->pixel[i] = malloc(img->width * pixelSize);
        fread(img->pixel[i], pixelSize, img->width, imageFile);
    }

    fclose(imageFile);
}

void writeImage(const char *outImgName, const image *img) {
    FILE *imageFile = fopen(outImgName, "w");

    // print image's type
    fprintf(imageFile, "P%c\n", img->type + '0');

    // print image's dimensions
    fprintf(imageFile, "%i %i\n%hhu\n", img->width, img->height, img->maxval);

    // print the pixel matrix
    int pixelSize = img->type == IMG_TYPE_COLOR ? 3 : 1;
    for (int i = 0; i < img->height; i++)
        fwrite(img->pixel[i], pixelSize, img->width, imageFile);

    fclose(imageFile);
}

void applyFilter(char *filterName, void **img, int imgType, int imgWidth, int numLines) {
    filter_type filter;
    if (!strcmp(filterName, "smooth"))
        filter = smoothFilter;
    else if (!strcmp(filterName, "blur"))
        filter = blurFilter;
    else if (!strcmp(filterName, "sharpen"))
        filter = sharpenFilter;
    else if (!strcmp(filterName, "mean"))
        filter = meanRemovalFilter;
    else if (!strcmp(filterName, "emboss"))
        filter = embossFilter;
    else
        return; // unknown filter

    // initialize auxiliary pixel matrix
    int pixelSize = imgType == IMG_TYPE_COLOR ? 3 : 1;
    void **imgAux = (void **) malloc((numLines + 1) * sizeof(void *));
    for (int i = 1; i <= numLines; i++)
        imgAux[i] = malloc(imgWidth * pixelSize);

    int i, j, l, c;
    for (i = 1; i <= numLines; i++) {
        // copy the ending pixels from the initial image
        if (imgType == IMG_TYPE_GRAYSCALE) {
            ((grayscale_pixel **) imgAux)[i][0] = ((grayscale_pixel **) img)[i][0];
            ((grayscale_pixel **) imgAux)[i][imgWidth - 1] = ((grayscale_pixel **) img)[i][imgWidth - 1];
        } else {
            ((color_pixel **) imgAux)[i][0] = ((color_pixel **) img)[i][0];
            ((color_pixel **) imgAux)[i][imgWidth - 1] = ((color_pixel **) img)[i][imgWidth - 1];
        }

        // apply filter on the others pixels
        for (j = 1; j < imgWidth - 1; j++) {
            if (imgType == IMG_TYPE_GRAYSCALE) {
                float bw = 0;
                for (l = -1; l <= 1; l++)
                    for (c = -1; c <= 1; c++)
                        bw += (filter.kernel[l + 1][c + 1] / filter.div) *
                              ((grayscale_pixel **) img)[i + l][j + c];

                ((grayscale_pixel **) imgAux)[i][j] = (unsigned char) bw;
            } else {
                float channel[3] = {0};
                for (l = -1; l <= 1; l++)
                    for (c = -1; c <= 1; c++)
                        for (int ch = 0; ch < 3; ch++)
                            channel[ch] += (filter.kernel[l + 1][c + 1] / filter.div) *
                                           ((color_pixel **) img)[i + l][j + c].channel[ch];

                for (int ch = 0; ch < 3; ch++)
                    ((color_pixel **) imgAux)[i][j].channel[ch] = (unsigned char) channel[ch];
            }
        }
    }

    // replace the old image with the filtered one
    for (i = 1; i <= numLines; i++) {
        free(img[i]);
        img[i] = imgAux[i];
    }
    free(imgAux);
}

/** Determine the range of lines which will be processed by the given process. */
bounds jobBoundsForProcess(int rank, int numLines, int procCount) {
    bounds jobBounds;
    int chunkSize = numLines / procCount;
    int r = numLines % procCount;

    jobBounds.begin = rank * chunkSize + (rank < r ? rank : r) + 1;
    jobBounds.end = jobBounds.begin + chunkSize + (rank < r ? 1 : 0);

    return jobBounds;
}

void freeImgMem(image *img) {
    for (int i = 0; i < img->height; i++)
        free(img->pixel[i]);
    free(img->pixel);
}

int main(int argc, char *argv[]) {
    int rank, procCount;
    MPI_Init(&argc, &argv);

    if (argc < 3) { // not enough arguments
        MPI_Finalize();
        return 0;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    image img;
    int imgType, imgHeight, imgWidth, numLines;
    void **pixelMat; // store the pixel lines assigned to the current process
    bounds jobBounds;
    MPI_Datatype pixelDataType;

    // create MPI type for color pixel struct
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint offsets[3] = {0, 1, 2};
    MPI_Datatype types[3] = {MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR};
    MPI_Datatype MPI_3UNSIGNED_CHAR;
    MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_3UNSIGNED_CHAR);
    MPI_Type_commit(&MPI_3UNSIGNED_CHAR);

    if (rank == 0) { // master process
        loadImage(argv[1], &img);

        jobBounds = jobBoundsForProcess(rank, img.height - 2, procCount);
        numLines = jobBounds.end - jobBounds.begin;
        pixelMat = img.pixel; // master process directly uses the pixel matrix from img
        pixelDataType = img.type == IMG_TYPE_COLOR ? MPI_3UNSIGNED_CHAR : MPI_UNSIGNED_CHAR;

        imgType = img.type;
        imgHeight = img.height;
        imgWidth = img.width;

        for (int i = 1; i < procCount; i++) {
            // send image properties to the other processes
            MPI_Send(&img.type, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&img.height, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&img.width, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

            // send pixel lines from the image
            bounds jobBounds = jobBoundsForProcess(i, img.height - 2, procCount);
            for (int line = jobBounds.begin - 1; line <= jobBounds.end; line++)
                MPI_Send(img.pixel[line], img.width, pixelDataType, i, 0, MPI_COMM_WORLD);
        }
    } else { // worker process
        MPI_Recv(&imgType, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
        MPI_Recv(&imgHeight, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
        MPI_Recv(&imgWidth, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);

        pixelDataType = imgType == IMG_TYPE_COLOR ? MPI_3UNSIGNED_CHAR : MPI_UNSIGNED_CHAR;
        int pixelSize = imgType == IMG_TYPE_COLOR ? 3 : 1;
        jobBounds = jobBoundsForProcess(rank, imgHeight - 2, procCount);
        numLines = jobBounds.end - jobBounds.begin;

        // receive the pixel lines to be processed
        pixelMat = (void **) malloc((numLines + 2) * sizeof(void *));
        for (int i = 0; i <= numLines + 1; i++) {
            pixelMat[i] = malloc(imgWidth * pixelSize);
            MPI_Recv(pixelMat[i], imgWidth, pixelDataType, 0, 0, MPI_COMM_WORLD, NULL);
        }
    }

    // apply all the filters before sending the final pixels to the master process
    for (int i = 3; i < argc; i++) {
        applyFilter(argv[i], pixelMat, imgType, imgWidth, numLines);

        // exchange bounding lines with the neighbor processes
        if (rank > 0)
            MPI_Sendrecv(pixelMat[1], imgWidth, pixelDataType, rank - 1, 0,
                         pixelMat[0], imgWidth, pixelDataType, rank - 1, 0, MPI_COMM_WORLD, NULL);

        if (rank < procCount - 1)
            MPI_Sendrecv(pixelMat[numLines], imgWidth, pixelDataType, rank + 1, 0,
                         pixelMat[numLines + 1], imgWidth, pixelDataType, rank + 1, 0, MPI_COMM_WORLD, NULL);
    }

    if (rank != 0) {
        // send final results to the master process
        for (int i = 1; i <= numLines; i++) {
            MPI_Send(pixelMat[i], imgWidth, pixelDataType, 0, 0, MPI_COMM_WORLD);
            free(pixelMat[i]);
        }
        free(pixelMat);
    } else {
        // retrieve and assembly the results
        for (int i = 1; i < procCount; i++) {
            bounds jobBounds = jobBoundsForProcess(i, img.height - 2, procCount);
            for (int line = jobBounds.begin; line < jobBounds.end; line++)
                MPI_Recv(img.pixel[line], img.width, pixelDataType, i, 0, MPI_COMM_WORLD, NULL);
        }

        writeImage(argv[2], &img);
        freeImgMem(&img);
    }

    MPI_Finalize();
    return 0;
}