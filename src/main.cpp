// =============================================================================
// Entry point.
//
// Compiles the starting expression, then hands control to the render loop.
// The expression may be given on the command line:
//
//     3DGraphingCalculator "sin(hypot(x, y) * 3 - t * 2)"
//
// A parse failure here is fatal and reported with the offending column, since
// there is no window yet to show it in. Once the app is running a bad
// expression is non-fatal: the previous surface keeps rendering and the error
// appears in the panel instead.
// =============================================================================

#include "GraphingCalculator3D.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    GraphingCalculator3D app;

    // Starting surface. Variables are x, y and t (seconds); type a new one
    // into the panel at runtime to replace it.
    const std::string source = (argc > 1) ? argv[1] : "sin(x - t) * cos(y - t)";

    std::string error;
    if (!app.setExpression(source, &error)) {
        std::cerr << "Cannot parse \"" << source << "\": " << error << "\n";
        return EXIT_FAILURE;
    }

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
