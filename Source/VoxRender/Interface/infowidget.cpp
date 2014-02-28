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

// Include Headers
#include "infowidget.h"
#include "ui_infowidget.h"

// Include Dependencies
#include "VoxScene/RenderController.h"
#include "VoxScene/FrameBuffer.h"
#include "VoxScene/Volume.h"
#include "mainwindow.h"

#define LEX_CAST(x) boost::lexical_cast<std::string>(x).c_str()

// -------------------------------------------------
//  Initializes the info widget UI
// -------------------------------------------------
InfoWidget::InfoWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::InfoWidget)
{
    ui->setupUi(this);
}
    
// -------------------------------------------------
//  Frees the info widget UI
// -------------------------------------------------
InfoWidget::~InfoWidget()
{
    delete ui;
}

// -------------------------------------------------
//  Sets the timing information in the tree view
// -------------------------------------------------
void InfoWidget::updatePerfStats(std::shared_ptr<vox::FrameBufferLock>)
{
    auto perfItem = ui->treeWidget->findItems( "Performance", Qt::MatchExactly ).front();
    
    auto timeItem = perfItem->child(0);
    timeItem->child(0)->setText(1, QString::number(MainWindow::instance->m_renderer->renderTime()));
    timeItem->child(1)->setText(1, QString::number(MainWindow::instance->m_renderer->tonemapTime()));
    timeItem->child(2)->setText(1, QString::number(MainWindow::instance->m_renderer->rndSeedTime()));
    
    perfItem->child(1);
    perfItem->child(2)->setText(1, QString::number(MainWindow::instance->m_renderController.iterations()));
    perfItem->child(3)->setText(1, QString::number(MainWindow::instance->m_renderController.renderTime()));

    ui->treeWidget->update();
}