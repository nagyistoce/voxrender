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

// VoxRender Includes
#include "VoxLib/Core/VoxRender.h"

// Include Dependencies
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

    // Connect scene change signal to the scene update slot :TODO: Change to recurring onFrame Callback and run on Xth cycles
    connect( MainWindow::instance, SIGNAL(sceneChanged( )), 
		this, SLOT(updateSceneStatistics( )) );
}
    
// -------------------------------------------------
//  Frees the info widget UI
// -------------------------------------------------
InfoWidget::~InfoWidget()
{
    delete ui;
}

// -------------------------------------------------
//  Sets the scene information in the tree view
// -------------------------------------------------
void InfoWidget::updateSceneStatistics( )
{
    // :TODO: Delay update until visibility //

    auto const& scene = MainWindow::instance->activeScene;

    auto sceneItem = ui->treeWidget->findItems( "Scene", Qt::MatchExactly ).front( );
    
        // Scene volume statistics
        auto volumeItem = sceneItem->child(0);
        auto const& volume = scene.volume.get( );
        if( volume )
        {
            volumeItem->child(0)->setText(1, "Filename.txt");
        }
        else
        {
            for( int i = 0; i < volumeItem->childCount( ); i++ )
                volumeItem->child(i)->setText(1, "");
        }

        // Camera statistics
        auto cameraItem = sceneItem->child(1);
        auto const& camera = scene.camera.get( );
        if (camera)
        {
			cameraItem->child(0)->setText(1, LEX_CAST(camera->filmWidth()));
            cameraItem->child(1)->setText(1, LEX_CAST(camera->filmHeight()));
			cameraItem->child(2)->setText(1, LEX_CAST(camera->apertureSize()));
			cameraItem->child(3)->setText(1, LEX_CAST(camera->focalDistance()));
            cameraItem->child(4)->setText(1, LEX_CAST(camera->fieldOfView( )));
        }
        else
        {
            for( int i = 0; i < cameraItem->childCount( ); i++ )
                volumeItem->child(i)->setText(1, "");
        }
}