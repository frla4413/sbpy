// Gmsh project created on Fri May  4 10:02:38 2018
SetFactory("OpenCASCADE");
//+
Point(1) = {-1, -1, 0, 1.0};
//+
Point(2) = {-1, 1, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {1, -1, 0, 1.0};
//+
Point(5) = {-0, 0, 0, 1.0};
//+
Point(6) = {-0.1, 0, 0, 1.0};
//+
Point(7) = {0, 0.1, 0, 1.0};
//+
Point(8) = {0.1, -0, 0, 1.0};
//+
Point(9) = {-0, -0.1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Line(5) = {1, 9};
//+
Line(6) = {2, 7};
//+
Recursive Delete {
  Curve{5}; 
}
//+
Recursive Delete {
  Curve{6}; 
}
//+
Recursive Delete {
  Point{6}; 
}
//+
Recursive Delete {
  Point{8}; 
}
//+
Point(6) = {-0.1, 0.1, 0, 1.0};
//+
Point(7) = {0.1, 0.1, 0, 1.0};
//+
Point(8) = {0.1, -0.1, 0, 1.0};
//+
Point(9) = {-0.1, -0.1, 0, 1.0};
//+
Line(5) = {1, 9};
//+
Line(6) = {2, 6};
//+
Line(7) = {3, 7};
//+
Line(8) = {4, 8};
//+
Circle(9) = {6, 5, 7};
//+
Circle(10) = {7, 5, 8};
//+
Circle(11) = {8, 5, 9};
//+
Circle(12) = {9, 5, 6};
//+
Curve Loop(1) = {1, 6, -12, -5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {6, 9, -7, -2};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {8, -10, -7, 3};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {4, 5, -11, -8};
//+
Plane Surface(4) = {4};
//+
Transfinite Surface {1} = {1, 2, 6, 9};
//+
Transfinite Surface {2} = {6, 2, 3, 7};
//+
Transfinite Surface {3} = {4, 8, 7, 3};
//+
Transfinite Surface {4} = {1, 9, 8, 4};
//+
Transfinite Curve {1}  = 161 Using Progression 1;
//+
Transfinite Curve {6}  = 161 Using Progression 1;
//+
Transfinite Curve {12} = 161 Using Progression 1;
//+
Transfinite Curve {5}  = 161 Using Progression 1;
//+
Transfinite Curve {2}  = 161 Using Progression 1;
//+
Transfinite Curve {7}  = 161 Using Progression 1;
//+
Transfinite Curve {9}  = 161 Using Progression 1;
//+
Transfinite Curve {3}  = 161 Using Progression 1;
//+
Transfinite Curve {10} = 161 Using Progression 1;
//+
Transfinite Curve {8}  = 161 Using Progression 1;
//+
Transfinite Curve {4}  = 161 Using Progression 1;
//+
Transfinite Curve {11} = 161 Using Progression 1;
//+
Recombine Surface {1, 2, 3, 4};
