// Gabriel Boroghina, 333CB

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

typedef struct {
    int div;
    int kernel[3][3];
} filter_type;

typedef struct {
    unsigned char channel[3]; // channel[0] = red, channel[1] = green, channel[2] = blue
}__attribute__((packed)) color_pixel;

typedef unsigned char grayscale_pixel; // light of the grayscale pixel

typedef enum {
    GRAYSCALE = 5,
    COLOR = 6
} image_type;

typedef struct {
    image_type type;
    int height, width;
    unsigned char maxval;
    void **pixel;
} image;

typedef struct {
    int begin, end;
} bounds;

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
    img->type = imageType;

    // read image's dimensions
    fscanf(imageFile, "%i%i%hhu%*c", &img->width, &img->height, &img->maxval);

    int pixelSize = img->type == COLOR ? 3 : 1;
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
    fprintf(imageFile, "P%c\n", img->type);

    // print image's dimensions
    fprintf(imageFile, "%i %i\n%hhu\n", img->width, img->height, img->maxval);

    // print the pixel matrix
    int pixelSize = img->type == COLOR ? 3 : 1;
    for (int i = 0; i < img->height; i++)
        fwrite(img->pixel[i], pixelSize, img->width, imageFile);

    fclose(imageFile);
}

void applyFilter(char *filterName, void **img, bounds lines) {
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


}

bounds jobBoundsForProcess(int rank, int numLines, int procCount) {
    bounds jobBounds;
    int chunkSize = numLines / procCount;
    int r = numLines % procCount;

    jobBounds.begin = rank * chunkSize + (rank < r ? rank : r) + 1;
    jobBounds.end = jobBounds.begin + chunkSize + (rank < r ? 1 : 0);

    return jobBounds;
}

int main(int argc, char *argv[]) {
    int rank, procCount;
    MPI_Init(&argc, &argv);

    if (argc < 3) {
        // not enough arguments
        MPI_Finalize();
        return 0;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    char *inImgName = argv[1], *outImgName = argv[2];
    image img;
    image_type imgType;
    int imgHeight, imgWidth, numLines;
    bounds jobBounds;
    void **imgBuf;
    MPI_Datatype pixelDataType;

    // create MPI type for color pixel struct
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR};
    MPI_Datatype MPI_3UNSIGNED_CHAR;
    MPI_Aint offsets[3] = {0, 1, 2};
    MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_3UNSIGNED_CHAR);
    MPI_Type_commit(&MPI_3UNSIGNED_CHAR);

    if (rank == 0) { // master process
        loadImage(inImgName, &img);

        jobBounds = jobBoundsForProcess(rank, img.height - 2, procCount);
        numLines = jobBounds.end - jobBounds.begin;
        imgBuf = img.pixel; // master process directly uses the pixel matrix
        pixelDataType = img.type == COLOR ? MPI_3UNSIGNED_CHAR : MPI_UNSIGNED_CHAR;

        imgType = img.type;
        imgHeight = img.height;
        imgWidth = img.width;

        // send basic info to the other processes
        for (int i = 1; i < procCount; i++) {
            MPI_Send(&img.type, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&img.height, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&img.width, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

            // send pixel lines from the image
            bounds jobBounds = jobBoundsForProcess(i, img.height - 2, procCount);
            printf("---- %i %i\n", jobBounds.begin, jobBounds.end);
            for (int line = jobBounds.begin - 1; line <= jobBounds.end; line++)
                MPI_Send(&img.pixel[line], img.width, pixelDataType, i, 0, MPI_COMM_WORLD);
        }
    } else { // worker process
        MPI_Recv(&imgType, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
        MPI_Recv(&imgHeight, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
        MPI_Recv(&imgWidth, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);

        pixelDataType = imgType == COLOR ? MPI_3UNSIGNED_CHAR : MPI_UNSIGNED_CHAR;
        jobBounds = jobBoundsForProcess(rank, imgHeight - 2, procCount);
        numLines = jobBounds.end - jobBounds.begin;
        int pixelSize = imgType == COLOR ? 3 : 1;

        imgBuf = (void **) malloc((numLines + 2) * sizeof(void *));
        for (int i = 0; i <= numLines + 1; i++) {
            imgBuf[i] = malloc(imgWidth * pixelSize);
            MPI_Recv(&imgBuf[i], imgWidth, pixelDataType, 0, 0, MPI_COMM_WORLD, NULL);
        }
    }

    // apply all the filters before sending the final pixel to the master process
    for (int i = 3; i < argc; i++) {
        applyFilter(argv[i], imgBuf, jobBounds);

        // exchange bounding lines with the neighbor processes
        if (rank > 0)
            MPI_Sendrecv(&imgBuf[1], imgWidth, pixelDataType, rank - 1, 0,
                         &imgBuf[0], imgWidth, pixelDataType, rank - 1, 0, MPI_COMM_WORLD, NULL);

        if (rank < procCount - 1)
            MPI_Sendrecv(&imgBuf[numLines], imgWidth, pixelDataType, rank + 1, 0,
                         &imgBuf[numLines + 1], imgWidth, pixelDataType, rank + 1, 0, MPI_COMM_WORLD, NULL);
    }

    if (rank != 0) {
        // send final results to the master process
        for (int i = 1; i <= numLines; i++)
            MPI_Send(&imgBuf[i], imgWidth, pixelDataType, 0, 0, MPI_COMM_WORLD);
    } else {
        // retrieve and assembly the results
        for (int i = 1; i < procCount; i++) {
            bounds jobBounds = jobBoundsForProcess(i, img.height - 2, procCount);
            for (int line = jobBounds.begin; line < jobBounds.end; line++)
                MPI_Recv(&img.pixel[line], img.width, pixelDataType, i, 0, MPI_COMM_WORLD, NULL);
        }

        writeImage(outImgName, &img);
    }

    MPI_Finalize();
    return 0;
}