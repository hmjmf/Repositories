<?php  
if ($_FILES["file"]["error"] > 0){  
	$code = $_FILES["file"]["error"];
	$msg = "FILE ERROR CODE: " . $_FILES["file"]["error"] . "<br />";  
	echo json_encode(array('code' => $code ,'msg' => $msg));
  
} else { 

	
	if ((($_FILES["file"]["type"] == "image/gif")
		|| ($_FILES["file"]["type"] == "image/jpeg")
		|| ($_FILES["file"]["type"] == "image/jpg")
		|| ($_FILES["file"]["type"] == "image/pjpeg")
		|| ($_FILES["file"]["type"] == "image/x-png")
		|| ($_FILES["file"]["type"] == "image/png"))){
		$original_path = "/tmp/" . $_FILES["file"]["name"];
		$extension = explode(".", $original_path);
		$result_path = "$original_path.result.${extension[count($extension) - 1]}"; 
		move_uploaded_file($_FILES["file"]["tmp_name"], $original_path);  

		system("./upload_image_demo $original_path $result_path" ,$callback);
		
		if($callback == 0){
			$code = 0;
			$msg = "success"; 
			
			echo json_encode(array('code' => $code ,'msg' => $msg,
						'original_path' =>  $original_path, 'result_path' =>  $result_path));
	 
		} else {
			$code = $callback;
			$msg = "API ERROR CODE: $callback"; 
			echo json_encode(array('code' => $code ,'msg' => $msg));
		}
			
	} else {
		$code = 1;
		$msg = "unexpect file type:" . $_FILES["file"]["type"]; 
		echo json_encode(array('code' => $code ,'msg' => $msg));
	}		
} 
?>