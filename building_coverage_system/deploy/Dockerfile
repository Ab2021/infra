# Multi-stage Dockerfile for Building Coverage System
# This Dockerfile creates a production-ready container for the building coverage system

# Build stage
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG APP_VERSION=1.0.0

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    gnupg2 \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && if [ "$BUILD_ENV" = "development" ]; then pip install -r requirements-dev.txt; fi

# Production stage
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production
ENV APP_VERSION=${APP_VERSION}

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    unixodbc \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directories
RUN mkdir -p /app /app/logs /app/data /app/config \
    && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories and set permissions
RUN mkdir -p /app/building_coverage_system/logs \
    && mkdir -p /app/building_coverage_system/data \
    && mkdir -p /app/building_coverage_system/config \
    && chown -R appuser:appuser /app

# Switch to application user
USER appuser

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import building_coverage_system; print('Health check passed')" || exit 1

# Default command
CMD ["python", "-m", "building_coverage_system.main"]

# Expose port (if running web service)
EXPOSE 8000

# Labels for metadata
LABEL maintainer="Building Coverage System Team"
LABEL version="${APP_VERSION}"
LABEL description="Building Coverage System - AI-powered insurance claim analysis"
LABEL org.opencontainers.image.source="https://github.com/company/building-coverage-system"