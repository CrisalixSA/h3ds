#!/bin/bash

# Constants
PACKAGE_NAME="h3ds"
MAIN_BRANCH="main"
VERSION_FILE_PATH="${PACKAGE_NAME}/__init__.py"
VERSION_IMPORT_PATH=$(echo ${VERSION_FILE_PATH%%.py} | tr '/' '.')

# Arguments
BUMP_PART=${1:-"minor"}     # Valid values: "major", "minor", "patch"

# Global variables
OLD_VERSION=$(python3 -c "from ${VERSION_IMPORT_PATH} import __version__; print(__version__)")
OLD_VERSION_MAJOR=$(echo ${OLD_VERSION} | cut -d '.' -f 1)
OLD_VERSION_MINOR=$(echo ${OLD_VERSION} | cut -d '.' -f 2)
OLD_VERSION_PATCH=$(echo ${OLD_VERSION} | cut -d '.' -f 3)
BRANCH_NAME="$(git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/\1/')"
BUMP_PART=$(echo ${BUMP_PART} | tr '[:upper:]' '[:lower:]')

# Check release branch
if [ "${BRANCH_NAME}" != "${MAIN_BRANCH}" ]; then
    echo "Releases have to be created at the ${MAIN_BRANCH} branch."
    echo "Please merge your changes to ${MAIN_BRANCH} if needed,"
    echo "checkout ${MAIN_BRANCH}, and re-launch this script."
    exit 1
fi

# Increment version
if [ "${BUMP_PART}" = "major" ]; then
    VERSION_MAJOR=$((${OLD_VERSION_MAJOR} + 1))
    VERSION_MINOR="0"
    VERSION_PATCH="0"
elif [ "${BUMP_PART}" = "patch" ]; then
    VERSION_MAJOR="${OLD_VERSION_MAJOR}"
    VERSION_MINOR="${OLD_VERSION_MINOR}"
    VERSION_PATCH=$((${OLD_VERSION_PATCH} + 1))
else
    VERSION_MAJOR="${OLD_VERSION_MAJOR}"
    VERSION_MINOR=$((${OLD_VERSION_MINOR} + 1))
    VERSION_PATCH="0"
fi
VERSION=${VERSION:-"${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"}

echo "Releasing version ${VERSION} (previous version: ${OLD_VERSION})"

# Update develop branch
git pull origin ${MAIN_BRANCH} || exit $?

# Update version file
cat "${VERSION_FILE_PATH}" | sed "s+${OLD_VERSION}+${VERSION}+g" > "/tmp/${PACKAGE_NAME}_version.py" || exit $?
cp -f "/tmp/${PACKAGE_NAME}_version.py" "${VERSION_FILE_PATH}" || exit $?

# Commit changes
git commit -a -m "Release v${VERSION}" || exit $?

# Push commit
git push origin "${MAIN_BRANCH}" --force-with-lease || exit $?

# Tag release
git tag -a "v${VERSION}" -m "Release v${VERSION}" || exit $?

# Push tag
git push origin "v${VERSION}" || exit $?

# Build pypi package
rm -rf h3ds.egg-info dist
python -m build

# Release pypi package
python -m twine upload dist/*

exit 0