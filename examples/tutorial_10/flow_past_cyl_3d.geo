SetFactory("OpenCASCADE");

// Create a variable to control mesh resolution

resolution_multiplier = 2.2;

// Create geometry

// Generate Box:
Point (100) = {0, 0, 0, 1.0};
Point (101) = {0, 0, 0.5, 1.0};
Point (102) = {0, 0, 1.5, 1.0};
Point (103) = {0, 0, 2.5, 1.0};

Point (104) = {0, 0.41, 0, 1.0};
Point (105) = {0, 0.41, 0.5, 1.0};
Point (106) = {0, 0.41, 1.5, 1.0};
Point (107) = {0, 0.41, 2.5, 1.0};

Point (108) = {0.41, 0.41, 0, 1.0};
Point (109) = {0.41, 0.41, 0.5, 1.0};
Point (110) = {0.41, 0.41, 1.5, 1.0};
Point (111) = {0.41, 0.41, 2.5, 1.0};

Point (112) = {0.41, 0, 0, 1.0};
Point (113) = {0.41, 0, 0.5, 1.0};
Point (114) = {0.41, 0, 1.5, 1.0};
Point (115) = {0.41, 0, 2.5, 1.0};

Line (100) = {100, 101};
Line (101) = {101, 102};
Line (102) = {102, 103};

Line (103) = {104, 105};
Line (104) = {105, 106};
Line (105) = {106, 107};

Line (106) = {108, 109};
Line (107) = {109, 110};
Line (108) = {110, 111};

Line (109) = {112, 113};
Line (110) = {113, 114};
Line (111) = {114, 115};

Line (112) = {100, 104};
Line (113) = {104, 108};
Line (114) = {108, 112};
Line (115) = {112, 100};

Line (116) = {103, 107};
Line (117) = {107, 111};
Line (118) = {111, 115};
Line (119) = {115, 103};

Curve Loop(1) = {107, 108, -117, -105, -104, -103, 113, 106};
Plane Surface(1) = {1};
Curve Loop(2) = {114, 109, 110, 111, -118, -108, -107, -106};
Plane Surface(2) = {2};
Curve Loop(3) = {110, 111, 119, -102, -101, -100, -115, 109};
Plane Surface(3) = {3};
Curve Loop(4) = {104, 105, -116, -102, -101, -100, 112, 103};
Plane Surface(4) = {4};
Curve Loop(5) = {114, 115, 112, 113};
Plane Surface(5) = {5};
Curve Loop(6) = {117, 118, 119, 116};
Plane Surface(6) = {6};
Surface Loop(1) = {4, 1, 2, 5, 3, 6};
Volume(1) = {1};

// Using Points, circle arcs, and lines to create the cylinder 
// (for better control of refinement around cylinder)
Point(9) = {-0.1, 0.2, 0.5, 0.01};
Point(10) = {-0.1, 0.25, 0.5, 0.01};
Point(11) = {-0.1, 0.2, 0.55, 0.01};
Point(12) = {-0.1, 0.15, 0.5, 0.01};
Point(13) = {-0.1, 0.2, 0.45, 0.01};
Point(14) = {0.61, 0.2, 0.5, 0.01};
Point(15) = {0.61, 0.25, 0.5, 0.01};
Point(16) = {0.61, 0.2, 0.55, 0.01};
Point(17) = {0.61, 0.15, 0.5, 0.01};
Point(18) = {0.61, 0.2, 0.45, 0.01};
Circle(13) = {10, 9, 11};
Circle(14) = {11, 9, 12};
Circle(15) = {12, 9, 13};
Circle(16) = {13, 9, 10};
Circle(17) = {15, 14, 16};
Circle(18) = {16, 14, 17};
Circle(19) = {17, 14, 18};
Circle(20) = {18, 14, 15};
Line(21) = {10, 15};
Line(22) = {11, 16};
Line(23) = {12, 17};
Line(24) = {13, 18};
// Closing the cylinder surfaces, and creating a cylinder volume:
Curve Loop(7) = {20, -21, -16, 24};
Surface(7) = {7};
Curve Loop(9) = {17, -22, -13, 21};
Surface(8) = {9};
Curve Loop(11) = {14, 23, -18, -22};
Surface(9) = {11};
Curve Loop(13) = {15, 24, -19, -23};
Surface(10) = {13};
Curve Loop(15) = {16, 13, 14, 15};
Surface(11) = {15};
Curve Loop(17) = {17, 18, 19, 20};
Surface(12) = {17};
Surface Loop(2) = {8, 12, 9, 11, 7, 10};
Volume(2) = {2};

// Using a boolean operator to cut the cylinder out of the box:
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }

// Mesh Refinement
Transfinite Curve {16, 18, 4, 19} = 2.5*resolution_multiplier Using Progression 1; // inlet
Transfinite Curve {17, 2, 24, 6} = 3*resolution_multiplier Using Progression 0.9; // pre-cylinder
Transfinite Curve {13, 1, 22, 8} = 8*resolution_multiplier Using Progression 1.05; // post-cylinder
Transfinite Curve {14, 3, 23, 7} = 3*resolution_multiplier Using Progression 1; // outflow
Transfinite Curve {5, 15, 20, 21} = 1*resolution_multiplier Using Progression 1; // outlet
Transfinite Curve {9, 12, 11, 10, 32, 31, 28, 27} = 1.5*resolution_multiplier Using Progression 1; // radial cyl
Transfinite Curve {30, 25, 26, 29} = 6*resolution_multiplier Using Progression 1; // longitudinal cyl

// Naming physical surfaces
Physical Surface("inlet", 30) = {3};
Physical Surface("outlet", 31) = {4};
Physical Surface("wall", 32) = {10, 5, 1, 2};
Physical Surface("cyl", 33) = {7, 8, 9, 6};
Physical Volume("volume", 34) = {1};
