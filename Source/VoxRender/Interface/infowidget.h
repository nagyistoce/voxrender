/* ===========================================================================

	Project: Info Widget - Rendering info display widget
    
	Description: Implements a display for render performance statistics.

    Copyright (C) 2012 Lucas Sherman

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

// Begin Definition
#ifndef INFO_WIDGET_H
#define INFO_WIDGET_H

// Include VoxRender
#include "VoxLib/Core/VoxRender.h"

// Include dependencies
#include <QtWidgets/QTreeWidget>
#include <QWidget>

namespace Ui 
{
	class InfoWidget;
}

// Render info display widget
class InfoWidget : public QWidget
{
    Q_OBJECT
    
public:
    explicit InfoWidget(QWidget *parent = 0);
    ~InfoWidget();

    void updatePerformanceStatistics();
    
public slots:
        void updatePerfStats(std::shared_ptr<vox::FrameBufferLock>);

private slots:
    void updateSceneStatistics();

private:
    Ui::InfoWidget *ui;
};

// End Definition
#endif // INFOWIDGET_H
