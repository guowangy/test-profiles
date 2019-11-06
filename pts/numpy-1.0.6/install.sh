#!/bin/sh

tar -xvf numpy-benchmarks-20160218.tar.gz
cd numpy-benchmarks-master/
PY_VERSION=$(python -c "import sys; print(sys.version_info[0])")
if [ $PY_VERSION -eq '3' ]
then  # Python 3 need patch
    patch -p1 <<'EOT'
diff -Nur a/benchit.py b/benchit.py
--- a/benchit.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchit.py	2019-11-06 14:35:21.170387675 +0800
@@ -2,6 +2,7 @@
 An adaptation from timeit that outputs some extra statistical informations
 '''
 
+from __future__ import print_function
 from timeit import default_timer, default_repeat, Timer
 import numpy
 import sys
@@ -27,9 +28,9 @@
         opts, args = getopt.getopt(args, "n:s:r:tcvh",
                                    ["number=", "setup=", "repeat=",
                                     "time", "clock", "verbose", "help"])
-    except getopt.error, err:
-        print err
-        print "use -h/--help for command line help"
+    except getopt.error as err:
+        print(err)
+        print("use -h/--help for command line help")
         return 2
     timer = default_timer
     stmt = "\n".join(args) or "pass"
@@ -56,7 +57,7 @@
                 precision += 1
             verbose += 1
         if o in ("-h", "--help"):
-            print __doc__,
+            print(__doc__)
             return 0
     setup = "\n".join(setup) or "pass"
     # Include the current directory, so that local imports work (sys.path
@@ -75,7 +76,7 @@
                 t.print_exc()
                 return 1
             if verbose:
-                print "%d loops -> %.*g secs" % (number, precision, x)
+                print("%d loops -> %.*g secs" % (number, precision, x))
             if x >= 0.2:
                 break
     try:
@@ -84,13 +85,13 @@
         t.print_exc()
         return 1
     if verbose:
-        print "raw times:", " ".join(["%.*g" % (precision, x) for x in r])
+        print("raw times:", " ".join(["%.*g" % (precision, x) for x in r]))
     r = [int(x * 1e6 / number) for x in r]
     best = min(r)
     average = int(numpy.average(r))
     std = int(numpy.std(r))
 
-    print best, average, std
+    print(best, average, std)
 
 if __name__ == "__main__":
     sys.exit(main())
diff -Nur a/benchmarks/allpairs_distances_loops.py b/benchmarks/allpairs_distances_loops.py
--- a/benchmarks/allpairs_distances_loops.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchmarks/allpairs_distances_loops.py	2019-11-06 14:35:21.150387675 +0800
@@ -2,6 +2,7 @@
 #run: allpairs_distances_loops(X, Y)
 
 #pythran export allpairs_distances_loops(float64[][], float64[][])
+from past.builtins import xrange
 import numpy as np
 
 def allpairs_distances_loops(X,Y):
diff -Nur a/benchmarks/conv.py b/benchmarks/conv.py
--- a/benchmarks/conv.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchmarks/conv.py	2019-11-06 14:35:21.150387675 +0800
@@ -2,6 +2,7 @@
 #run: conv(x,w)
 
 #pythran export conv(float[][], float[][])
+from past.builtins import xrange
 import numpy as np
 
 def clamp(i, offset, maxval):
@@ -22,6 +23,6 @@
         for j in xrange(sx[1]):
             for ii in xrange(sw[0]):
                 for jj in xrange(sw[1]):
-                    idx = clamp(i,ii-sw[0]/2,sw[0]), clamp(j,jj-sw[0]/2,sw[0])
+                    idx = clamp(i,ii-int(sw[0]/2),sw[0]), clamp(j,jj-int(sw[0]/2),sw[0])
                     result[i,j] += x[idx] * weights[ii,jj]
     return result
diff -Nur a/benchmarks/diffusion.py b/benchmarks/diffusion.py
--- a/benchmarks/diffusion.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchmarks/diffusion.py	2019-11-06 14:35:21.150387675 +0800
@@ -1,4 +1,4 @@
-#setup: import numpy as np;lx,ly=(2**7,2**7);u=np.zeros([lx,ly],dtype=np.double);u[lx/2,ly/2]=1000.0;tempU=np.zeros([lx,ly],dtype=np.double)
+#setup: import numpy as np;lx,ly=(2**7,2**7);u=np.zeros([lx,ly],dtype=np.double);u[int(lx/2),int(ly/2)]=1000.0;tempU=np.zeros([lx,ly],dtype=np.double)
 #run: diffusion(u,tempU,100)
 
 #pythran export diffusion(float [][], float [][], int)
diff -Nur a/benchmarks/fft.py b/benchmarks/fft.py
--- a/benchmarks/fft.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchmarks/fft.py	2019-11-06 14:35:21.150387675 +0800
@@ -2,7 +2,7 @@
 #run: fft(a)
 
 #pythran export fft(complex [])
-
+from past.builtins import xrange
 import math, numpy as np
 
 def fft(x):
diff -Nur a/benchmarks/grayscott.py b/benchmarks/grayscott.py
--- a/benchmarks/grayscott.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchmarks/grayscott.py	2019-11-06 14:35:21.150387675 +0800
@@ -12,8 +12,8 @@
 
     r = 20
     u[:] = 1.0
-    U[n/2-r:n/2+r,n/2-r:n/2+r] = 0.50
-    V[n/2-r:n/2+r,n/2-r:n/2+r] = 0.25
+    U[int(n/2)-r:int(n/2)+r,int(n/2)-r:int(n/2)+r] = 0.50
+    V[int(n/2)-r:int(n/2)+r,int(n/2)-r:int(n/2)+r] = 0.25
     u += 0.15*np.random.random((n,n))
     v += 0.15*np.random.random((n,n))
 
diff -Nur a/benchmarks/growcut.py b/benchmarks/growcut.py
--- a/benchmarks/growcut.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchmarks/growcut.py	2019-11-06 14:35:21.150387675 +0800
@@ -3,6 +3,7 @@
 #run: growcut(image, state, state_next, 10)
 
 #pythran export growcut(float[][][], float[][][], float[][][], int)
+from past.builtins import xrange
 import math
 import numpy as np
 def window_floor(idx, radius):
diff -Nur a/benchmarks/hyantes.py b/benchmarks/hyantes.py
--- a/benchmarks/hyantes.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchmarks/hyantes.py	2019-11-06 14:35:21.150387675 +0800
@@ -1,4 +1,4 @@
-#setup: import numpy ; a = numpy.array([ [i/10., i/10., i/20.] for i in xrange(80)], dtype=numpy.double)
+#setup: import numpy ; from past.builtins import xrange ; a = numpy.array([ [i/10., i/10., i/20.] for i in xrange(80)], dtype=numpy.double)
 #run: hyantes(0, 0, 90, 90, 1, 100, 80, 80, a)
 
 #pythran export hyantes(float, float, float, float, float, float, int, int, float[][])
diff -Nur a/benchmarks/local_maxima.py b/benchmarks/local_maxima.py
--- a/benchmarks/local_maxima.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchmarks/local_maxima.py	2019-11-06 14:35:21.150387675 +0800
@@ -22,6 +22,6 @@
   for pos in np.ndindex(data.shape):
     myval = data[pos]
     for offset in np.ndindex(wsize):
-      neighbor_idx = tuple(mode(p, o-w/2, w) for (p, o, w) in zip(pos, offset, wsize))
+      neighbor_idx = tuple(mode(p, o-int(w/2), w) for (p, o, w) in zip(pos, offset, wsize))
       result[pos] &= (data[neighbor_idx] <= myval)
   return result
diff -Nur a/benchmarks/pairwise.py b/benchmarks/pairwise.py
--- a/benchmarks/pairwise.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchmarks/pairwise.py	2019-11-06 14:35:21.150387675 +0800
@@ -3,6 +3,7 @@
 #run: pairwise(X)
 
 #pythran export pairwise(float [][])
+from past.builtins import xrange
 
 import numpy as np
 def pairwise(X):
diff -Nur a/benchmarks/smoothing.py b/benchmarks/smoothing.py
--- a/benchmarks/smoothing.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchmarks/smoothing.py	2019-11-06 14:35:21.150387675 +0800
@@ -3,6 +3,7 @@
 #from: http://www.parakeetpython.com/
 
 #pythran export smoothing(float[], float)
+from past.builtins import xrange
 
 def smoothing(x, alpha):
   """
diff -Nur a/benchmarks/wdist.py b/benchmarks/wdist.py
--- a/benchmarks/wdist.py	2016-02-18 22:21:28.000000000 +0800
+++ b/benchmarks/wdist.py	2019-11-06 14:35:21.150387675 +0800
@@ -3,6 +3,7 @@
 #run: wdist(A,B,W)
 
 #pythran export wdist(float64 [][], float64 [][], float64[][])
+from past.builtins import xrange
 
 import numpy as np
 def wdist(A, B, W):
EOT
fi
cd ~

echo $? > ~/install-exit-status
echo "#!/bin/sh
cd numpy-benchmarks-master/
python2 run.py -t python > numpy_log
echo 'Test name :   Avg time ( nanoseconds )'
cat numpy_log | awk 'BEGIN{total_avg_time=0} {print \$1\":\"\$4;total_avg_time+=\$4;} END{printf(\"\n\n-------------------\nTotal avg time (nanoseconds): %.02f\n\", total_avg_time);}' \$@ > \$LOG_FILE
echo \$? > ~/test-exit-status " > numpy
chmod +x numpy
