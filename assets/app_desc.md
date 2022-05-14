This is a simple project that implements a content-based image retrieval engine using PostgreSQL as storage backend and Python for the application logic. PostgreSQL database contains a sample of images metadata extracted from cat and dogs images of [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset. 

With this application you can upload your own image and search for similar images contained in the PostgreSQL database. To do so you have to provide number K of nearest neighbor images to retrieve and the distance metric to use for the searching.

[**Github repository**](https://github.com/ZisisFl/content-based-image-search)