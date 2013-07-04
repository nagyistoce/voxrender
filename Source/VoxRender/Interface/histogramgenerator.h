/* ===========================================================================

	Project: VoxRender
    
	Description: Generates the histogram images for the volume data

    Copyright (C) 2013 Lucas Sherman

	Lucas Sherman, email: LucasASherman@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================== */

// Begin definition
#ifndef HISTOGRAM_GENERATOR_H
#define HISTOGRAM_GENERATOR_H

// Include Dependencies
#include "mainwindow.h"
#include "histogramview.h"

// VoxLib Dependencies
#include "VoxLib/Core/Geometry/Color.h"
#include "VoxLib/Core/Geometry/Image.h"

// QT Dependencies
#include <QtCore/QEvent>

class HistogramGenerator : public QObject
{
    Q_OBJECT
    
public:
    static HistogramGenerator * instance()
    {
        static HistogramGenerator instance;

        return &instance;
    }

    ~HistogramGenerator();

    QPixmap & densityPixmap()  { return m_densityPixmap;  }
    QPixmap & gradientPixmap() { return m_gradientPixmap; }
    QPixmap & laplacePixmap()  { return m_laplacePixmap;  }

public slots:
    void generateHistogramImages();

    void stopGeneratingImages();

signals:
    void histogramComplete(int dataType);

    void histogramImageReady(int dataType);

private slots:
    void onHistogramImageReady(int dataType);

private:
    HistogramGenerator();

    void threadEntryPoint(std::shared_ptr<vox::Volume> volume);

    void generateDensityHistogram(std::shared_ptr<vox::Volume> volume);

    void generateGradientHistogram(std::shared_ptr<vox::Volume> volume);
        
    void generateLaplacianHistogram(std::shared_ptr<vox::Volume> volume);

    std::shared_ptr<boost::thread> m_thread;

    QPixmap m_densityPixmap;
    QPixmap m_gradientPixmap;
    QPixmap m_laplacePixmap;

    vox::Image<vox::ColorRgbaLdr> m_densityImage;
    vox::Image<vox::ColorRgbaLdr> m_gradientImage;
    vox::Image<vox::ColorRgbaLdr> m_laplaceImage;
};

// End definition
#endif // HISTOGRAM_GENERATOR_H